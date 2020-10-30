import tensorflow as tf
import time
from seq2seq_pgn_tf2.utils.losses import loss_function
from seq2seq_pgn_tf2.models.pgn import PGN
from seq2seq_pgn_tf2.test_helper import beam_decode
from seq2seq_pgn_tf2.batcher import batcher, Vocab
from rouge import Rouge
from tqdm import tqdm

def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])
    # open validation y_data
    with open(params["valid_seg_y_dir"],'r') as f:
        valid_targets=f.readlines()
        valid_targets = valid_targets[:params['num_to_valid']]
    val_params = params.copy()
    rouge = Rouge()
    # @tf.function()
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask):
        # loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            outputs = model(enc_output,  # shape=(3, 200, 256)
                            dec_hidden,  # shape=(3, 256)
                            enc_inp,  # shape=(3, 200)
                            enc_extended_inp,  # shape=(3, 200)
                            dec_inp,  # shape=(3, 50)
                            batch_oov_len,  # shape=()
                            enc_padding_mask,  # shape=(3, 200)
                            params['is_coverage'],
                            prev_coverage=None)
            loss = loss_function(dec_tar,
                                 outputs,
                                 padding_mask,
                                 params["cov_loss_wt"],
                                 params['is_coverage'])
        
        # variables = model.trainable_variables
        variables = model.encoder.trainable_variables +\
                    model.attention.trainable_variables +\
                    model.decoder.trainable_variables +\
                    model.pointer.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
        for batch in dataset:
            loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
                              batch[0]["extended_enc_input"],  # shape=(16, 200)
                              batch[1]["dec_input"],  # shape=(16, 50)
                              batch[1]["dec_target"],  # shape=(16, 50)
                              batch[0]["max_oov_len"],  # ()
                              batch[0]["sample_encoder_pad_mask"],  # shape=(16, 200)
                              batch[1]["sample_decoder_pad_mask"])  # shape=(16, 50)
            step += 1
            total_loss += loss
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))
    results = valid(val_params)
    scores_avg = rouge.get_scores(results, valid_targets, avg=True)
    print('validated average rouge score is:',scores_avg)

def valid(params):
    params['mode'] = 'test'
    params['batch_size'] = params['beam_size']

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    params['test_seg_x_dir']= params["valid_seg_x_dir"]
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params["pgn_model_dir"])
    ckpt = tf.train.Checkpoint(PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # path = params["model_path"] if params["model_path"] else ckpt_manager.latest_checkpoint
    # path = ckpt_manager.latest_checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    results=[]
    b_iter = iter(b)
    print("start to validate*********")
    for _ in tqdm(range(params['num_to_valid'])):
        batch = next(b_iter)
        results.append(beam_decode(model, batch, vocab, params).abstract)
    return results