import numpy as np
import json
import random
from collections import deque
from env import *
from config import *
import tensorflow as tf
import json
import sys
import os
import time

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
EPISODE = 20000
STEP = 5

class DQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.good_buffer =  {} 
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.feat_dim
        self.action_op_dim = 3
        self.create_Q_network()
        self.create_training_method()
        self.count = 0

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        self.state_input = tf.placeholder("float",[None,self.state_dim])

        W1 = self.weight_variable([self.state_dim,50])
        b1 = self.bias_variable([50])
        h_layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        W2 = self.weight_variable([50, 50])
        b2 = self.bias_variable([50])
        h_layer_2 = tf.nn.relu(tf.matmul(h_layer_1, W2) + b2)

        W_action_op = self.weight_variable([50, self.action_op_dim])
        b_action_op = self.bias_variable([self.action_op_dim])

        self.Q_op_value = tf.matmul(h_layer_2, W_action_op) + b_action_op

    def create_training_method(self):
        self.action_op_input = tf.placeholder("float",[None,self.action_op_dim]) # one hot presentation
        self.y_op_input = tf.placeholder("float",[None])
        self.Q_op_action = tf.reduce_sum(tf.mul(self.Q_op_value,self.action_op_input),reduction_indices = 1)
        self.op_cost = tf.reduce_mean(tf.square(self.y_op_input - self.Q_op_action))

        self.op_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.op_cost)

    def perceive(self,state, action_op,reward,next_state,done, step):
        self.count += 1
        one_hot_op_action = np.zeros(self.action_op_dim)
        one_hot_op_action[action_op] = 1
        if reward > 0 :
            self.good_buffer[(step,reward)] = (state,one_hot_op_action,reward,next_state,done, step)
        if self.count % 10000 == 0:
            self.count = 0
            for k,v in self.good_buffer.iteritems():
                self.replay_buffer.append(v) 
                if len(self.replay_buffer) > REPLAY_SIZE:
                    self.replay_buffer.popleft()
        else:
            self.replay_buffer.append((state,one_hot_op_action,reward,next_state,done, step))
            if len(self.replay_buffer) > REPLAY_SIZE:
                self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def egreedy_action(self,state):
        Q_op_value = self.Q_op_value.eval(feed_dict = {
            self.state_input:np.array([state])
            })[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_op_dim - 1)
        else:
            return np.argmax(Q_op_value)
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        else:
            self.epsilon = FINAL_EPSILON

    def action(self,state):
        Q_op_value = self.Q_op_value.eval(feed_dict = {
            self.state_input:np.array([state])
            })[0]
        return np.argmax(Q_op_value)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_op_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]


        # Step 2: calculate y
        y_op_batch = []
        Q_op_value_batch = self.Q_op_value.eval(feed_dict={self.state_input:next_state_batch})
        #print "Q_value_batch:", Q_value_batch
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_op_batch.append(reward_batch[i])
            else :
                y_op_batch.append(reward_batch[i] + GAMMA * np.max(Q_op_value_batch[i]))

        #print y_batch
        #print self.Q_action.eval(feed_dict={self.action_input:action_batch, self.state_input:state_batch})
        #print self.cost.eval(feed_dict = {self.y_input:y_batch, self.action_input:action_batch,self.state_input:state_batch})
        self.op_loss = self.op_cost.eval(feed_dict={
            self.y_op_input:y_op_batch,
            self.action_op_input:action_op_batch,
            self.state_input:state_batch
        })
        print("operate_loss", self.op_loss)
        self.op_optimizer.run(feed_dict={
            self.y_op_input:y_op_batch,
            self.action_op_input:action_op_batch,
            self.state_input:state_batch
        })


def main():
    config = Config()
    config.ana_filename = config.ana_filename + "_" + sys.argv[1]
    config.train_list, config.validate_list = config.seperate_date_set(sys.argv[1])
    env = Env(config)
    env.make_env()
    dqn = DQN(env)
    checkpoint_dir = "./model/fold" + sys.argv[1]
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver = tf.train.Saver()
    saver.restore(dqn.session, latest_checkpoint) 
    right_count = 0
    total_time = 0
    for itera in xrange(config.validate_num):
        state = env.vali_reset(itera)
        for step in xrange(STEP):
             action_op = dqn.action(state)
             next_state,done,flag,etime = env.val_step(action_op, sys.argv[1])
             total_time += etime
             state = next_state
             if done:
                 right_count += flag
                 if flag == 1:
                     print('---\t'+str(config.validate_list[itera]))
                 break
        #print "test_index:", config.validate_list[itera], "reward", total_reward
    print("predict time:", total_time) 
    print('Evaluation Average Accuracy:' , right_count*1.0/config.validate_num)

if __name__ == '__main__':
    main()  
