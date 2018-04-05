from numpy.random import randint
import tensorflow as tf

def sanity_data_maker(in_file, out_file, n_examples=50, n_lines=3):
    indices = randint(0, 2000, size=(n_examples,))
    l_i = 0
    selected_lines = [str(k) for k in range(22-n_lines, 22)]
    sum_len, sum_q_len = 0, 0
    
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        for line in fin:
            if l_i > 2000:
                break
            splitted = line.split()
            
            if splitted and splitted[0] in selected_lines and l_i in indices:
                sum_len += len(splitted) - 1
                fout.write(line)
            
            if splitted and splitted[0] == '21':
                if l_i in indices:
                    sum_q_len += len(splitted) - 1
                    fout.write('\n')
                l_i += 1

    return sum_len / n_examples, sum_q_len / n_examples

def val_accuracy(sess, data_loader, acc_tensor, params, abatch_size=100):
    total_acc = 0
    for val_step in range(0, data_loader.val_data_len, abatch_size):
        batch_val = data_loader.sample_batch('val', abatch_size, offset=val_step)

        val_dict = {params[0]:batch_val[0],params[1]:batch_val[1],
                       params[2]:batch_val[3],params[3]:batch_val[4],params[4]:batch_val[5]}
        iacc = sess.run(acc_tensor, feed_dict=val_dict)
        total_acc += iacc * batch_val[0].shape[0]
    return total_acc / data_loader.val_data_len

def val_cands_accuracy(sess, data_loader, a_hat, params, abatch_size=100):
    total_acc = 0
    for val_step in range(0, data_loader.val_data_len, abatch_size):
        batch_val = data_loader.sample_batch('val', abatch_size, offset=val_step)

        val_dict = {params[0]:batch_val[0],params[1]:batch_val[1],
                       params[2]:batch_val[3],params[3]:batch_val[4],params[4]:batch_val[5]}
        c_hat = sess.run(a_hat, feed_dict=val_dict)
        
        for i in range(abatch_size):
            cands = batch_val[2][i]
            scores = c_hat[i, cands]
            prediction = cands[np.argmax(scores)]
            total_acc += (prediction == batch_val[3][i])
        
    return total_acc / data_loader.val_data_len

def val_predictions(sess, data_loader, params, abatch_size=100):
    all_preds = []
    for val_step in range(0, data_loader.val_data_len, abatch_size):
        batch_val = data_loader.sample_batch('val', abatch_size, offset=val_step)

        val_dict = {params[0]:batch_val[0],params[1]:batch_val[1],
                       params[2]:batch_val[3],params[3]:batch_val[4],params[4]:batch_val[5]}
        c_preds = sess.run(a_hat, feed_dict=val_dict)
        answs = np.argmax(c_preds, axis=1)
        all_preds += list(answs)
    return np.array(all_preds)

def masked_mean(X, mask, axis=0, keepdims=True, name=None):
    with tf.name_scope(name, 'mean', [X]):
        norm = tf.reduce_sum(mask, axis, keepdims=keepdims)
        return tf.reduce_sum(X * mask, axis, keepdims=keepdims) / norm

def masked_softmax(X, mask, axis=0, EPS=1e-15, name=None):
    with tf.name_scope(name, 'softmax', [X]):
        max_axis = tf.reduce_max(X, axis, keepdims=True)
        X_exp = tf.exp(X - max_axis) * mask
        norm = tf.reduce_sum(X_exp, axis, keepdims=True)
        return X_exp / (norm + EPS)
