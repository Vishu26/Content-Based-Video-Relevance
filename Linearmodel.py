import tensorflow as tf
import utils
tf.logging.set_verbosity(tf.logging.ERROR)

curr_index = '0'
curr_value = '1'

W_1 = tf.get_variable(name='W_1', shape=[512, 256])
b_1 = tf.get_variable(name='b_1', initializer=tf.constant(0.0))

W_2 = tf.get_variable(name='W_2', shape=[256, 64])
b_2 = tf.get_variable(name='b_2', initializer=tf.constant(0.0))

W_3 = tf.get_variable(name='W_3', shape=[64, 1])
b_3 = tf.get_variable(name='b_3', initializer=tf.constant(0.0))

rel_list = utils.load_relevance('shows/track_1_shows/relevance_train.csv')
idx = utils.split_index('shows/track_1_shows/split/train.csv')
da = utils.load_pool('shows/track_1_shows/feature', idx)
print(rel_list[curr_index])


X = tf.placeholder(shape=[1,512], dtype=tf.float32 ,name='X')
print('HI')
y1 = tf.matmul(X, W_1) + b_1
y2 = tf.matmul(y1, W_2) + b_2
y = tf.matmul(y2, W_3) + b_3

def loss():
	loss = 0
	l = len(rel_list[curr_index])

	sec_score = tf.matmul(da[curr_value].reshape(1, 512), W_1) + b_1
	sec_score = tf.matmul(sec_score, W_2) + b_2
	sec_score = tf.matmul(sec_score, W_3) + b_3
	if curr_value in rel_list[curr_index]:
		ind = rel_list[curr_index].index(curr_value)

		abs_diff = tf.abs(y - sec_score)
		loss = (l-ind)*10*abs_diff

	else:

		abs_diff = tf.abs(y - sec_score)
		loss = (10 / abs_diff)
	return loss

lo = loss()

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(lo)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	#global curr_index

	sess.run(init)
	for i in range(5):
		for j in da:
			curr_index = j
			for k in da:
				if j!=k:

					curr_value = k
					_, los = sess.run([optimizer, lo], feed_dict={X:da[j].reshape(1, 512)})
					if curr_value in rel_list[curr_index]:
						print(los)

	with open('log.txt', 'w') as f:


		for j in da:

			f.write(j+' '+str((sess.run(y, feed_dict={X:da[j].reshape(1, 512)}))[0][0])+'\n')

