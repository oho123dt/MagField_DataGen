import pickle
import os
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Add, Concatenate, Activation
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

### BEFORE RUNNING THE CODE
### DOWNLOAD THE DATASET AND WEIGHT FILE AT DRIVE LINK:
### https://drive.google.com/drive/folders/1mStGTjXSFSzdJys8VSxpooajJegmnx7Q?usp=sharing

dir = 'Data_and_model/Data_and_model/EC_data_4m/'
os.chdir(dir)

with open('re_s_y_train_scaled_az', 'rb') as filename:
  s_y_train_scaled = pickle.load(filename)

with open('re_s_x_test_scaled_az', 'rb') as filename:
  s_x_test_scaled = pickle.load(filename)

with open('re_s_y_test_scaled_az', 'rb') as filename:
  s_y_test_scaled = pickle.load(filename)

with open('re_y_sScaler_az', 'rb') as filename:
  s_y_sScaler = pickle.load(filename)

with open('re_x_sScaler_az', 'rb') as filename:
  s_x_sScaler = pickle.load(filename)

with open('EC_para_200k_outtest', 'rb') as filename:
  s_x_outtest = pickle.load(filename)

with open('para_changing_r_feature', 'rb') as filename:
  s_x_test_r = pickle.load(filename)

with open('output_field_az_changing_r_feature', 'rb') as filename:
  s_y_test_r = pickle.load(filename)

with open('para_changing_z_feature', 'rb') as filename:
  s_x_test_z = pickle.load(filename)

with open('output_field_az_changing_z_feature', 'rb') as filename:
  s_y_test_z = pickle.load(filename)

with open('EC_output_field_200k_outtest', 'rb') as filename:
  s_y_outtest = pickle.load(filename)
  outtest = []
for elem in s_y_outtest:
  outtest.append([elem[1]])
s_y_outtest = np.array(outtest)
print(s_y_outtest.shape)

s_y_train = s_y_sScaler.inverse_transform(s_y_train_scaled)
s_y_test = s_y_sScaler.inverse_transform(s_y_test_scaled)
s_x_outtest_scaled = s_x_sScaler.transform(s_x_outtest)
s_x_test_r_scaled = s_x_sScaler.transform(s_x_test_r)
s_x_test_z_scaled = s_x_sScaler.transform(s_x_test_z)
def NLL(y_true, y_pred):
  return -y_pred.log_prob(y_true) 

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(   
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = Sequential(
        [
            tfpl.VariableLayer(
                tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfpl.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def PBNN_model_init():
  model = Sequential([
        tfpl.DenseVariational (
                  units = 796,
                  make_prior_fn = prior, make_posterior_fn = posterior,
                  kl_weight = 1 / int(len(s_x_train_scaled)/2) , activation = 'sigmoid'),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(796, activation=tf.nn.relu),
        Dense(units = tfpl.MultivariateNormalTriL.params_size(1)),
        tfpl.MultivariateNormalTriL(1)   
  ])
  return model

def PNN_model_init():
  model = Sequential([
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(796, activation=tf.nn.relu),
          Dense(units = tfpl.MultivariateNormalTriL.params_size(1)),
          tfpl.MultivariateNormalTriL(1)   
  ])
  return model

model = PBNN_model_init()

opt = tf.keras.optimizers.Adam()

model.compile(optimizer=opt, loss=NLL)

weight_file = 'path-to-weight-file'
model = model_init()
y = model(s_x_test_r_scaled)
model.load_weights(weight_file)
s_y_r_predicted = model(s_x_test_r_scaled)
s_y_z_predicted = model(s_x_test_z_scaled)

Z = []
r = []
for elem in s_x_test_z:
  Z.append(elem[5])
for elem in s_x_test_r:
  r.append(elem[3])

y_hat = s_y_z_predicted.mean()
y_sd = s_y_z_predicted.stddev()
y_hat_m2sd = y_hat - 2 * y_sd
y_hat_p2sd = y_hat + 2 * y_sd

plt.plot(Z, s_y_test_z, color = 'blue', label = 'ground truth')
plt.plot(Z, y_hat, color = 'red', alpha = 1, label = 'model $\mu$', lw = 2)
plt.plot(Z, y_hat_m2sd, color = 'green', alpha = 0.7, label = 'model $\mu \pm 2 \sigma$')
plt.plot(Z, y_hat_p2sd, color = 'green', alpha = 0.7)
plt.title('Distribution versus change of Z feature')
plt.legend()
plt.show()

z_index = 666
plt.plot(Z[z_index:], s_y_test_z[z_index:], color = 'blue', label = 'ground truth')
plt.plot(Z[z_index:], y_hat[z_index:], color = 'red', alpha = 1, label = 'model $\mu$', lw = 2)
plt.plot(Z[z_index:], y_hat_m2sd[z_index:], color = 'green', alpha = 0.7, label = 'model $\mu \pm 2 \sigma$')
plt.plot(Z[z_index:], y_hat_p2sd[z_index:], color = 'green', alpha = 0.7)
plt.title('Close look at out-range Z-feature and Mag. field')
plt.legend()
plt.show()

r_index = 750
y_hat = s_y_r_predicted.mean()
y_sd = s_y_r_predicted.stddev()
y_hat_m2sd = y_hat - 2 * y_sd
y_hat_p2sd = y_hat + 2 * y_sd
plt.plot(r, s_y_test_r, color = 'blue', label = 'ground truth')
plt.plot(r, y_hat, color = 'red', alpha = 1, label = 'model $\mu$', lw = 2)
plt.plot(r, y_hat_m2sd, color = 'green', alpha = 0.7, label = 'model $\mu \pm 2 \sigma$')
plt.plot(r, y_hat_p2sd, color = 'green', alpha = 0.7)
plt.title('Distribution versus change of r feature')
plt.legend()
plt.show()

plt.plot(r[r_index:], s_y_test_r[r_index:], color = 'blue', label = 'ground truth')
plt.plot(r[r_index:], y_hat[r_index:], color = 'red', alpha = 1, label = 'model $\mu$', lw = 2)
plt.plot(r[r_index:], y_hat_m2sd[r_index:], color = 'green', alpha = 0.7, label = 'model $\mu \pm 2 \sigma$')
plt.plot(r[r_index:], y_hat_p2sd[r_index:], color = 'green', alpha = 0.7)
plt.title('Close look at out-range r-feature and Mag. field')
plt.legend()
plt.show()
