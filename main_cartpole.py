# Sunjeet Jena| 17:23, Friday, 27th July, 2018
# This code is for Training and Testing the "Cartpole-v0" problem in Open Gym
# By default the code is written for training it in GPU and thus requires tensorflow gpu
# As of now dated 27th July, 2018 only Deep Q-Network has been coded for training the problem


# Importing all the required libraries
import gym
import tensorflow as tf
import numpy as np 	
import time
from colorama import Fore, Back, Style
from collections import deque
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5' 
video_foldername="Training and Testing Videos for Mountain_Car_Discrete" # Name of the folder to store the training and testing videos

Mini_batch=32						#Batch Size over which gradient would be evaluated
max_episodes=3000					#Number of Episodes in each epoch
max_steps=200						#Maximum number of steps the agent can take in each episode
Number_of_Epochs=100				#Maximum number of epochs for training
Discount_Factor=0.99				#Discount factor for estimation of future reward in the Network
Replay_Memory=10000					#Max Size of Replay Memory
Epsilon_decay=0.0005				#Epsilon decay
Epsilon_Step=1000					#Step till to use Epsilon Greedy
Weight_Update_Step_Size=20			#Updatation of weights from Evaluation Network to target network
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon

sess=tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))	#Creating the Tensorflow session 


class Network():
	# Sunjeet Jena| 17:45, Friday, 27th July, 2018
	# The deep learning network for predicting the Q-Values given the state
	# This network has one layer

	def __init__(self, input, scope_name, Batch_Size=1):

		# 'input' is the state input given to the network to predict the Q-Values
		# 'scope_name' is the name of the scope under which the network will predict. Example: Target Network or The Evaluation network
		# 'Batch_Size' is size of the batch to be evaluated
		
		self.input=input		# Keeping the input in the class 				
		

		with tf.variable_scope(scope_name) as scope:	#Creating the Scope

			#with tf.variable_scope("Fully_Connected_Layer_1"):	#Creating the Scope for Output layer 

			self.first_fully_connected_1=tf.layers.dense(inputs=self.input, units=50,activation=tf.nn.relu,kernel_initializer= tf.truncated_normal_initializer (),
											 bias_initializer=tf.initializers.ones(),name="Fully_Connected_Layer_1")	#Fully Connected Operation
			
			
			with tf.variable_scope("Fully_Connected_Layer_2"):	#Creating the Scope for Output layer 

				self.first_fully_connected_2=tf.layers.dense(inputs=self.first_fully_connected_1, units=50,activation=tf.nn.relu,kernel_initializer= tf.truncated_normal_initializer (),
															 bias_initializer=tf.initializers.ones(),name="Fully_Connected_Layer_2")	#Fully Connected Operation
			
			"""
			with tf.variable_scope("Fully_Connected_Layer_3"):	#Creating the Scope for Output layer 

				self.first_fully_connected_3=tf.layers.dense(inputs=self.first_fully_connected_2, units=256,kernel_initializer= tf.truncated_normal_initializer (),name="Fully_Connected_Layer_3")	#Fully Connected Operation
			"""
			#with tf.variable_scope("Output_Layer"):	#Creating the Scope for Output layer 

			self.output_layer=tf.layers.dense(inputs=self.first_fully_connected_2, units=2,kernel_initializer= tf.truncated_normal_initializer (), 
												 bias_initializer=tf.initializers.ones(),name="Output_Layer")	#Fully Connected Operation
			#Note there is no activation function in the operation. By default tensorflow uses linear activation function
			
			self.Max_Q_Values=tf.reduce_max(self.output_layer, axis=-1)		# Max Q-Values predicted for each given state 
			self.Action_Output=tf.argmax(self.output_layer, axis=-1, name='Output_Action_as_Given_by_Deep_Networks')	#Output as given network for each individual state
		
			self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)	#Getting all the variables under the given scope
			self.trainable_vars_by_name = {var.name[len(scope.name):]: var for var in self.trainable_vars}	# Storing all the variables by scope name


	def Q_Values_of_Given_State_Action(self, actions_, y_targets):

			# Sunjeet Jena| 18:32, Friday, 27th July, 2018
			# This function is for calculating the reward given the action and the target values as given by addition of reward and predicted Q-Value
			
			self.output_layer=self.output_layer 	#Getting the Q-Values from the output layer
			actions_=tf.reshape(tf.cast(actions_, tf.int32), shape=(Mini_batch,1))	#Casting action to int32 type and Reshaping the input action array
			z=tf.reshape(tf.range(tf.shape(self.output_layer)[0]), shape=(Mini_batch,1) )	#Creating the index as the tf.gather_nd takes the following format[Array number, index number]
			index_=tf.concat((z,actions_), axis=-1)	#Getting the index values to produce the the Q-Values
			self.Q_Values_Select_Actions=tf.gather_nd(self.output_layer, index_)	#Producing the Q-Values from the given indices
			#loss_1=tf.divide((tf.reduce_sum (tf.square(self.Q_Values_Select_Actions-y_targets))), 2)	#Calculating the loss
			loss=tf.reduce_mean(tf.square(self.Q_Values_Select_Actions-y_targets))
			
			return loss



with tf.device('/device:GPU:0'):
	# Sunjeet Jena| 18:32, Friday, 27th July, 2018
	#Creating scope for placing the required variables in the GPU
	
	# All the variables declared under are for target network

	x_Target_Values=tf.placeholder(dtype=tf.float32,shape=(None,4), name="x_Target_Values") 		# Creating place holder for giving state as input to the network
	Network_Object_Target=Network(x_Target_Values, 'Target_Network')								# Creating the Object for Networks class for Target Networks
	Target_Network_Q_Values_=Network_Object_Target.output_layer 									# Getting the Q-Values of all action of each state from the Q-Values
	Target_Network_Max_Q_Values_=Network_Object_Target.Max_Q_Values 								# Getting the Max Q-Values given the state
	target_vars = Network_Object_Target.trainable_vars_by_name										# Getting the Trainable parameters in the target network

	########################################################

	# All the variables declared under are for Evaluation network

	x_Eval_Net=tf.placeholder(dtype=tf.float32,shape=(None,4), name="x_Eval_Net") 		# Creating place holder for giving state as input to the network
	a_t_eval=tf.placeholder(dtype=tf.int32,shape=(Mini_batch), name="a_t_eval")			# Creating the placeholder to input action in given state as stored in the replay memory
	y_Targets_eval=tf.placeholder(dtype=tf.float32,shape=(Mini_batch), name="y_Targets_eval")	#Creating the placeholder to input Target values as produced by adding reward and Q-values of next state from target netwrok
	Network_Object_Evaluation=Network(x_Eval_Net, 'Evaluation_Network')					# Creating the Object for Networks class for Evaluation network
	Eval_Network_Q_Values_=Network_Object_Evaluation.output_layer 							# Getting the Q-Values each action of each state from the Q-Values
	Eval_Network_Max_Q_Values_=Network_Object_Evaluation.Max_Q_Values 							# Getting the Max Q-Values
	eval_vars = Network_Object_Evaluation.trainable_vars_by_name						# Getting the Trainable parameters in the evaluation network
	#########################################################
	# Training the Network

	loss=Network_Object_Evaluation.Q_Values_of_Given_State_Action(a_t_eval,y_Targets_eval)	#Calculating the loss
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
	train_= optimizer.minimize(loss, var_list=tf.trainable_variables(scope='Evaluation_Network'))

	##########################################################
	copy_ops =[target_var.assign(eval_vars[var_name]) for var_name, target_var in target_vars.items()]	#Copying the variables
	copy_online_to_target = tf.group(*copy_ops)					
	
	######################
	init_op = tf.global_variables_initializer()		#Initializing the global variables
	sess.run(init_op)


def random_samples(Beta_Set, Mini_batch):
	# Sunjeet Jena| 14:27, Saturday, 28th July, 2018
	# This Function takes the beta set and generates random samples for training
	# 'Mini_batch' is the size of the batch to be trained

	dataset_state=[]	#Dataset to store the initial states of the randomly selected samples
	dataset_action=[]	#Dataset to store the action taken of the randomly selected samples
	dataset_reward=[]	#Dataset to store the reward from the action of the randomly selected samples
	dataset_state_plus_1=[]	#Dataset to store the next states of the randomly selected samples

	for i in range(Mini_batch):	#Loopin over the minibatch size
		s=np.random.randint(len(Beta_Set))	#Getting a random integer from the uniform distribution
		sample_=Beta_Set[s]					#Getting the random sample
		dataset_state.append(sample_[0])		#Storing the initial state of the random sample
		dataset_action.append(sample_[1])	#Storing the action of the random sample
		dataset_reward.append(sample_[2])	#Storing the reward of the random sample
		dataset_state_plus_1.append(sample_[3])	#Storing the final state of the random sample

	return	(dataset_state,dataset_action,dataset_reward,dataset_state_plus_1)	#Returning the final samples


def Target_Values(rewards_train,states_plus_1_train):	#For DQN

	# Sunjeet Jena| 14:50, Saturday, 28th July, 2018
	# This function is for generating the target values given st+1 state and reward
	# Q-Values of the next state are generated and added with the reward to get the target values

	#with tf.variable_scope('Target_Network') as scope:

	Output_Q_Values=sess.run((Target_Network_Max_Q_Values_), feed_dict={x_Target_Values:states_plus_1_train})	#Getting the Max Q-Values
	y_target_values=(Discount_Factor*np.asarray(Output_Q_Values)) +rewards_train		#Adding the reward and the Q-value along with the discount factor 				

	return y_target_values


def Evaluation_Network(states_train,actions_train, Targets):

	# Sunjeet Jena| 14:54, Saturday, 28th July, 2018
	# This function takes the states, the actions and the targets and performs a gradient descent on evaluation network
	#with tf.variable_scope('Eval_Network') as scope:

	
	loss_,_=sess.run((loss,train_), feed_dict={x_Eval_Net:states_train, a_t_eval:actions_train,y_Targets_eval:Targets})	#Feeds the network and trains the system	
	#print(loss_)
	return loss_ 

def decay_function(steps):

	# Sunjeet Jena| 00:37, Sunday, 29th July, 2018
	# This function is for epsilon decay 
	epsilon= -steps*Epsilon_decay+1

	if(epsilon<0.1):
		epsilon=0.1

	return epsilon


def DQN(states_train, actions_train, rewards_train, states_plus_1_train):
	# Sunjeet Jena| 14:47, Saturday, 28th July, 2018
	# This function is Deep Q-Network training function

	Targets=Target_Values(rewards_train,states_plus_1_train)	#Getting the target values from the target network using target function
	Eval=Evaluation_Network(states_train,actions_train, Targets)# Perform gradient descent on the evaluation network and obtain the batch loss
	return Eval 	#Return the loss

def main():

	# Sunjeet Jena| 13:11, Saturday, 28th July, 2018
	# This is the main function where we use Open AI Gym for getting the observation, action and reward and training the architecture
	# There are two parts here one is training and the other is testing

	#Training

	Beta_Set=deque(maxlen=Replay_Memory)	# Initializing the replay memory list
	env = gym.make('CartPole-v0').env		# Making the environment for MountainCar-v0 from Open AI Gym
	copy_online_to_target.run(session=sess)

	Global_Steps=0			#Initializing the global step size to keep a track of total number of steps taken	

	#epsilon=INITIAL_EPSILON

	for e in range(Number_of_Epochs):	#Looping for all epochs

		#print('IN EPOCH NUMBER: ' + str(e))
		for episode in range(max_episodes):	#Looping it over all episodes

			print('In Episode Number : ' + str(episode))
			observation = env.reset()	#Resetting the Environment and getting the observation

			reward_this_episode=0
			done=False

			train_counts=0	#Counter to keep the training count in the episode
			episodic_loss=0	#Counter to keep the episodic loss
			for step in range(max_steps):	#Looping it till maximum steps
				
				env.render()				#Rendering the Environment 
				Q_=sess.run((Eval_Network_Q_Values_),feed_dict={x_Eval_Net:[observation]})	#Feeding the Observation to the network and getting the Q Values

				state_=observation 	#Storing the initial state of the sample
				
				epsilon=decay_function(Global_Steps)
				if(epsilon>=np.random.uniform()): #Condition Check for checking epsilon greedy exploration 
					action = env.action_space.sample()
				else:
					action=np.argmax(Q_)

				observation, reward, done, info = env.step(action)
				Global_Steps=Global_Steps+1
				reward_this_episode=reward_this_episode+reward
				state_plus_1=observation 	#Storing the final state of the sample
				
				if done and reward_this_episode<200 :
					reward = -500   # If it fails, punish hard
				
				sample=[state_, action, reward, state_plus_1]	#Generating the sample
				Beta_Set.append(sample)	#Adding the sample to the replay memory

				if(len(Beta_Set)>Mini_batch):
					states_train, actions_train, rewards_train, states_plus_1_train=random_samples(Beta_Set, Mini_batch)	#Getting the random samples
					train=DQN(states_train, actions_train, rewards_train, states_plus_1_train)		#Training the given the mini batch sample and obtain the batch loss
					
					train_counts=train_counts+1
					episodic_loss=episodic_loss+train
				
				if(done==True):
					break

			if(episode%Weight_Update_Step_Size==0 and episode!=0):	#Condition Check for Weight Updation
				copy_online_to_target.run(session=sess) #Copying the weights
				print(Style.BRIGHT+Fore.WHITE+ Back.BLACK+'Weight Updated'+Style.RESET_ALL)

			
			if(len(Beta_Set)>Mini_batch):			
				print('Average Episodic Loss: ' + str(episodic_loss/train_counts)+ '\n')

			TEST=10
			STEP=200
			if episode % 100 == 0:
				total_reward = 0
				for i in range(TEST):
					print('Testing in loop: '+ str(i))
					state = env.reset()
					for j in range(STEP):
						env.render()
						Q_=sess.run((Eval_Network_Q_Values_),feed_dict={x_Eval_Net:[state]})
						action=np.argmax(Q_)
						state,reward,done,_ = env.step(action)
						total_reward += reward
						if done:
							break
				ave_reward = total_reward/TEST
				print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
			#env.close()			

	copy_online_to_target.run(session=sess)	
	print(Style.BRIGHT+Fore.WHITE+ Back.BLACK+'Weight Updated'+Style.RESET_ALL)

	#Testing		

	while (1):

		observation = env.reset()	#Resetting the Environment and getting the observation
		while(1) :	
			env.render()	#Rendering the Environment 
			Q_=sess.run((Eval_Network_Q_Values_),feed_dict={x_Eval_Net:[observation]})	#Feeding the Observation to the network and getting the Q Values
			action=np.argmax(Q_)	
			observation, reward, done, info = env.step(action)
			print(action)
			if(done==True):
				print('Done')
				break
main()