"""The authors' code, copied unchanged from seRNN_demo.ipynb in their CodeOcean capsule
(Achterberg et al. 2023, capsule 2879348, version 2).

Contents:
  mazeGeneratorI           - the maze task
  SE1                      - the distance-only wiring cost
  SE1_sWc                  - the seRNN wiring cost (distance x communicability)
  RNNWeightMatrixHistoryI  - records the recurrent weights after every epoch

Do not edit: the point of the replication is to use their code as published.
Only the import lines below were added, taken from the same notebook.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import Regularizer
from tensorflow.python.keras import backend
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance
import matplotlib.pyplot as plt


class mazeGeneratorI():
    '''
    Objects of the mazeGeneratorI class can create numpy and tf datasets of the first choice of the maze task.
    Task structure:
        Goal presentation, followed by delay period, followed by choice options.
    Response:
        One response required from agent at end of episode. Direction (Left, Up, Right, Down) of first step.
    Encoding:
        Both observations and labels are OneHot encoded.
    Usage:
        The two only function a user should need to access are "construct_numpy_data" and "construct_tf_data"
    Options:
        Both data construction methods have an option to shuffle the labels of data.
        The numpy data construction method allows to also return the maze identifiers.
    '''
    def __init__(self, goal_presentation_steps, delay_steps, choices_presentation_steps):
        self.version = 'v1.2.0'
        
        # Import variables defining episode
        self.goal_presentation_steps = goal_presentation_steps
        self.delay_steps = delay_steps
        self.choices_presentation_steps = choices_presentation_steps

        # Construct mazes dataframe
        ## Add encoded versions of the goal / choices presentations and the next step response
        self.mazesdf = self.import_maze_dic()
        self.mazesdf['Goal_Presentation'] = self.mazesdf['goal'].map({
            7:np.concatenate((np.array([1,0,0,0]),np.repeat(0,4))),
            9:np.concatenate((np.array([0,1,0,0]),np.repeat(0,4))),
            17:np.concatenate((np.array([0,0,1,0]),np.repeat(0,4))),
            19:np.concatenate((np.array([0,0,0,1]),np.repeat(0,4)))})
        self.mazesdf['Choices_Presentation']=self.mazesdf['ChoicesCategory'].map(lambda x: self.encode_choices(x=x))
        self.mazesdf['Step_Encoded']=self.mazesdf['NextFPmap'].map(lambda x: self.encode_next_step(x=x))

    def construct_numpy_data(self, number_of_problems, return_maze_identifiers = False, np_shuffle_data = False):
        # Create a new column which hold the vector for each problem
        self.mazesdf['Problem_Vec']=self.mazesdf.apply(lambda x: self.create_problem_observation(row= x,goal_presentation_steps= self.goal_presentation_steps,delay_steps= self.delay_steps,choices_presentation_steps= self.choices_presentation_steps), axis=1)
        # Set a random order of maze problems for the current session
        self.mazes_order = np.random.randint(0,8,number_of_problems)

        # Create vectors, holding observations and labels
        session_observation =np.array([])
        session_labels = np.array([])
        for i in self.mazes_order:
            session_observation = np.append(session_observation,self.mazesdf.iloc[i]['Problem_Vec'])
            session_labels = np.append(session_labels,self.mazesdf.iloc[i]['Step_Encoded'])

        # Reshape vectors to fit network observation and response space
        session_length = self.goal_presentation_steps + self.delay_steps + self.choices_presentation_steps
        session_observation = np.reshape(session_observation, (-1,session_length,8)).astype('float32')
        session_labels = np.reshape(session_labels, (-1,4)).astype('float32')

        # If np_shuffle_data == 'Labels, the order of labels is shuffled to randomise correct answers
        if np_shuffle_data == 'Labels':
          shuffle_generator = np.random.default_rng(38446)
          shuffle_generator.shuffle(session_labels,axis=0)

        # If return_maze_identifiers == 'IDs', return the array with maze IDs alongside the regular returns (observations, labels)
        if return_maze_identifiers == 'IDs':
          return session_observation, session_labels, self.mazes_order

        return session_observation, session_labels

    def construct_tf_data(self, number_of_problems, batch_size, tf_shuffle_data = False):
        # Create dataset as described by numpy dataset function and transform it into a TF dataset
        npds, np_labels = self.construct_numpy_data(number_of_problems=number_of_problems, np_shuffle_data = tf_shuffle_data)
        tfdf = tf.data.Dataset.from_tensor_slices((npds, np_labels))
        tfdf = tfdf.batch(batch_size)
        return tfdf

    def reset_construction_params(self, goal_presentation_steps, delay_steps, choices_presentation_steps):
        self.goal_presentation_steps = goal_presentation_steps
        self.delay_steps = delay_steps
        self.choices_presentation_steps = choices_presentation_steps

    def import_maze_dic(self, mazeDic=None):
        if mazeDic == None:
            # Set up dataframe with first choices of maze task
            ## The dictionary was generated using MazeMetadata.py (v1.0.0) and the following call:
            ### mazes.loc[(mazes['Nsteps']==2)&(mazes['ChoiceNo']=='ChoiceI')][['goal','ChoicesCategory','NextFPmap']].reset_index(drop=True).to_dict()
            self.mazesDic = {'goal': {0: 9, 1: 9, 2: 19, 3: 17, 4: 17, 5: 7, 6: 19, 7: 7},
            'ChoicesCategory': {0: 'ul',
            1: 'rd',
            2: 'ld',
            3: 'rd',
            4: 'ul',
            5: 'ur',
            6: 'lr',
            7: 'lr'},
            'NextFPmap': {0: 'u', 1: 'r', 2: 'd', 3: 'd', 4: 'l', 5: 'u', 6: 'r', 7: 'l'}}
        else:
            self.mazesDic = mazesDic
        
        # Create and return dataframe
        return pd.DataFrame(self.mazesDic)

    def encode_choices(self, x):
        # Helper function to create the observation vector for choice periods
        choices_sec = np.repeat(0,4)
        choicesEncoding = pd.Series(list(x))
        choicesEncoding = choicesEncoding.map({'l':1,'u':2,'r':3,'d':4})
        for encodedChoice in choicesEncoding:
            choices_sec[encodedChoice-1]=1
        return np.concatenate((np.repeat(0,4),choices_sec))

    def encode_next_step(self, x):
        # Helper function to change the response / action to a OneHot encoded vector
        step_sec = np.repeat(0,4)
        stepEncoding = pd.Series(list(x))
        stepEncoding = stepEncoding.map({'l':1,'u':2,'r':3,'d':4})
        for encodedStep in stepEncoding:
            step_sec[encodedStep-1]=1
        return step_sec

    def create_problem_observation(self, row, goal_presentation_steps, delay_steps, choices_presentation_steps):
        # Helper function to create one vector describing the entire outline of one maze problem (Goal presentation, Delay Period, and Choices Presentation)
        goal_vec = np.tile(row['Goal_Presentation'], goal_presentation_steps)
        delay_vec = np.tile(np.repeat(0,8), delay_steps)
        choices_vec = np.tile(row['Choices_Presentation'], choices_presentation_steps)
        problem_vec = np.concatenate((goal_vec,delay_vec,choices_vec))
        return problem_vec

    def __repr__(self):
        return '\n'.join([
            f'Maze DataSet Generator',
            f'Goal Presentation Steps: {self.goal_presentation_steps}',
            f'Delay Steps: {self.delay_steps}',
            f'Choices Presentation Steps: {self.choices_presentation_steps}'])



class SE1(Regularizer):
  """A regulariser for sptially embedded RNNs.
  Applies L1 regularisation to recurrent kernel of
  RNN which is weighted by the distance of units
  in predefined 3D space.
  Calculation:
      se1 * sum[distance_matrix o recurrent_kernel]
  Attributes:
      se1: Float; Weighting of SE1 regularisation term.
      distance_tensor: TF tensor / matrix with cost per
      connection in weight matrix of network.
  """

  def __init__(self, se1=0.01, neuron_num = 100, network_structure = (5,5,4), coordinates_list = None, distance_power = 1, distance_metric = 'euclidean'):  
    self.version = 'v1.1.0'
    self.distance_power = distance_power
    
    # Set SE1 regularisation strength to default of 0.01 if no value given
    se1 = 0.01 if se1 is None else se1
    self._check_penalty_number(se1)

    # Transform regularisation strength to TF's standard float format 
    self.se1 = backend.cast_to_floatx(se1)

    # Set up tensor with distance matrix
    ## Set up neurons per dimension
    nx = np.arange(network_structure[0])
    ny = np.arange(network_structure[1])
    nz = np.arange(network_structure[2])

    ## Set up coordinate grid
    [x,y,z] = np.meshgrid(nx,ny,nz)
    self.coordinates = [x.ravel(),y.ravel(),z.ravel()]

    ## Override coordinate grid if one if provided in init
    if coordinates_list!=None:
      self.coordinates = coordinates_list

    ## Check neuron number / number of coordinates
    if (len(self.coordinates[0])==neuron_num)&(len(self.coordinates[1])==neuron_num)&(len(self.coordinates[2])==neuron_num):
      pass
    else:
      raise ValueError('Network / coordinate structure does not match the number of neurons.')

    ## Calculate the euclidean distance matrix
    euclidean_vector = scipy.spatial.distance.pdist(np.transpose(self.coordinates), metric=distance_metric)
    euclidean = scipy.spatial.distance.squareform(euclidean_vector**self.distance_power)
    self.distance_matrix = euclidean.astype('float32')
    self.spatial_cost_matrix = self.distance_matrix

    ## Add minimal cost for recurrent self connection (on diagonal)
    #diag_dist = np.diag(np.repeat(0.1,100)).astype('float32')
    #self.distance_matrix = self.distance_matrix + diag_dist

    ## Create tensor from distance matrix
    self.distance_tensor =  tf.convert_to_tensor(self.distance_matrix)


  def __call__(self, x):
    # Add calculation of loss here.
    # L1 for reference: self.l1 * math_ops.reduce_sum(math_ops.abs(x))
    abs_weight_matrix = tf.math.abs(x)

    #se1_loss = self.se1 * tf.math.multiply(abs_weight_matrix, self.distance_tensor)
    #se1_loss = tf.math.reduce_sum(abs_weight_matrix)
    se1_loss = self.se1 * tf.math.reduce_sum(tf.math.multiply(abs_weight_matrix, self.distance_tensor))
    
    return se1_loss

  def _check_penalty_number(self, x):
    """check penalty number availability, raise ValueError if failed."""
    if not isinstance(x, (float, int)):
      raise ValueError(('Value: {} is not a valid regularization penalty number, '
                        'expected an int or float value').format(x))

  def visualise_distance_matrix(self):
    plt.imshow(self.distance_matrix)
    plt.colorbar()
    plt.show()

  def visualise_neuron_structure(self):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(self.coordinates[0],self.coordinates[1],self.coordinates[2],c='b',marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

  def get_config(self):
    return {'se1': float(self.se1)}


class SE1_sWc(SE1):
    '''
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    '''
    def __init__(self, se1=0.01, comms_factor = 1, neuron_num = 100, network_structure = (5,5,4), coordinates_list = None, distance_power = 1, distance_metric = 'euclidean'):
      SE1.__init__(self, se1, neuron_num , network_structure , coordinates_list, distance_power , distance_metric)
      self.comms_factor = comms_factor

    def __call__(self, x):
      # Take absolute of weights
      abs_weight_matrix = tf.math.abs(x)

      # Calulcate weighted communicability (see reference in docstring)
      stepI = tf.math.reduce_sum(abs_weight_matrix, axis=1)
      stepII = tf.math.pow(stepI, -0.5)
      stepIII = tf.linalg.diag(stepII)
      stepIV = tf.linalg.expm(stepIII@abs_weight_matrix@stepIII)
      comms_matrix = tf.linalg.set_diag(stepIV, tf.zeros(stepIV.shape[0:-1]))

      # Multiply absolute weights with communicability weights
      comms_matrix = comms_matrix**self.comms_factor
      comms_weight_matrix = tf.math.multiply(abs_weight_matrix, comms_matrix)

      # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
      se1_loss = self.se1 * tf.math.reduce_sum(tf.math.multiply(comms_weight_matrix , self.distance_tensor))
      
      return se1_loss


class RNNWeightMatrixHistoryI(keras.callbacks.Callback):
    '''
    Saves the RNN_Weight_Matrix to the training history before
    the start of training and after finishing each epoch.

    The network model needs to be build manually before calling fit() method
    for this callback to work.
    '''
    def __init__(self, RNN_layer_number = 0):
        super(RNNWeightMatrixHistoryI, self).__init__()
        self.RNN_layer_number = RNN_layer_number

    def on_train_begin(self, logs=None):
        # Create key for RNN_Weight_Matrix in history
        self.model.history.history['RNN_Weight_Matrix'] = []
        #print("Created key for RNN_Weight_Matrix in history.")

        # Save matrix before start of training
        self.model.history.history['RNN_Weight_Matrix'].append(self.model.layers[self.RNN_layer_number].get_weights()[1])
        #print("Saved RNN_Weight_Matrix to history.")

    def on_epoch_end(self, epoch, logs=None):
        # Save RNN_Weight_Matrix to history
        self.model.history.history['RNN_Weight_Matrix'].append(self.model.layers[self.RNN_layer_number].get_weights()[1])
        #print("Saved RNN_Weight_Matrix to history.")
