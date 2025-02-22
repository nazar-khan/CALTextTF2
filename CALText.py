import argparse
import tensorflow as tf
import utility
import data
import random
import time
from CALTextModel import Model
from compute_error import compute_cer_wer
import numpy as np

# Directories
results_folder = 'results/'
model_folder = 'model/'
checkpoints_folder = 'checkpoints/'
data_folder = 'data/'
log_dir = "./logs/"
log_file = "training_log.txt"

# Constants
MAXIMAGESIZE = 500000
BATCH_IMAGESIZE = 500000
BATCHSIZE = 2
MAX_LEN = 130
BEAM_SIZE = 3
NUM_EPOCHS = 50
PATIENCE = 15  # Early stopping patience
LR_DECAY_FACTOR = 0.5  # Reduce LR by 50% if no improvement
MIN_LR = 1e-6  # Minimum learning rate

def main(args):

    # Dataset paths
    data_folder_path = f"{data_folder}{args.dataset}/"
    
    # Load training, validation, and test data
    train, train_uid_list = data.dataIterator(data_folder_path + 'train_lines.pkl',
                                              data_folder_path + 'train_labels.pkl',
                                              batch_size=BATCHSIZE, batch_Imagesize=BATCH_IMAGESIZE, 
                                              maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
    valid, valid_uid_list = data.dataIterator(data_folder_path + 'valid_lines.pkl',
                                              data_folder_path + 'valid_labels.pkl',
                                              batch_size=BATCHSIZE, batch_Imagesize=BATCH_IMAGESIZE, 
                                              maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
    test, test_uid_list = data.dataIterator(data_folder_path + 'test_lines.pkl',
                                            data_folder_path + 'test_labels.pkl',
                                            batch_size=1, batch_Imagesize=BATCH_IMAGESIZE, 
                                            maxlen=MAX_LEN, maxImagesize=MAXIMAGESIZE)
    
    # For checking to see if everything works correctly
    #train = list(train)[:2*BATCHSIZE]  # Use only the first 10 batches
    #valid = list(valid)[:2*BATCHSIZE]   # Use only the first 5 batches
    #test = list(test)[:2*BATCHSIZE]     # Use only the first 5 batches
    
    # Load vocabulary
    worddicts, vocabulary_count = utility.load_dict_picklefile(data_folder_path + 'vocabulary_unicode.pkl')
    eol_index = worddicts['\n'][0]	#EOL is the newline character which is 10 in Unicode
    space_index = worddicts[' '][0]
    #print("wordicts: ", worddicts)
    #print("eol_index: ", eol_index)
    #print("space_index: ", space_index)
    num_classes = vocabulary_count# + 1
    #print("num_classes: ", num_classes)
    
    '''
    # Load the combined vocabulary from pickle files
    vocab_mapping = utility.load_combined_vocabulary(data_folder_path + "vocabulary.pkl", data_folder_path + "vocabulary_unicode.pkl")
    
    # Print the mapping
    for index, (char, unicode_char) in sorted(vocab_mapping.items()):
        print(f"Index: {index}, Character: {char}, Unicode Character: {unicode_char}")
    '''

    # Initialize model
    model = Model(num_classes, eol_index)
    model.alpha_reg.assign(args.alpha_reg)  # Set alpha_reg from command-line argument
    
    # Add an epoch counter variable
    epoch_counter = tf.Variable(0, dtype=tf.int64)

    # Setup checkpointing
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model, epoch=epoch_counter)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoints_folder, max_to_keep=5)
    

    # TensorBoard logging
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Check if a checkpoint exists
    if checkpoint_manager.latest_checkpoint:
        #print(f"Restoring from {checkpoint_manager.latest_checkpoint}")
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        start_epoch = int(epoch_counter.numpy())  # Convert to Python integer
        print(f"Restored from {checkpoint_manager.latest_checkpoint} at epoch {start_epoch}")
    else:
        start_epoch = 0
        # Logging setup
        with open(log_file, "w") as f:
            f.write("Epoch\tTrain Loss\tValid Loss\tLR\n")
        print("No checkpoint found, starting from scratch.")
    
    compute_error_rates = 0

    if args.mode == "train":
        best_val_loss = float('inf')
        patience_counter = 0
        #for epoch in range(NUM_EPOCHS):
        for epoch in range(start_epoch, NUM_EPOCHS):
            start_time = time.time()
            random.shuffle(train)
            
            # Training loop
            total_loss = 0
            batch_num = 0
            for batch_x, batch_y in train:
                batch_num += 1
                batch_x, batch_x_m, batch_y, batch_y_m = data.prepare_data(batch_x, batch_y)
                #print("batch_x.shape: ", batch_x.shape)
                #print("batch_y.shape: ", batch_y.shape)
                loss = model.train_step(batch_x, batch_y, batch_x_m, batch_y_m)
                total_loss += loss
                print(f"Epoch {epoch + 1}, Batch {batch_num}, Batch Loss: {loss:.4f}, Training Loss: {total_loss/batch_num:.4f}")

            avg_train_loss = total_loss / len(train)
            print(f"\nEpoch {epoch + 1}: Training Loss = {avg_train_loss:.4f}")
            
            # Update epoch counter
            epoch_counter.assign(epoch + 1)

            # Validation loop
            total_val_loss = 0
            batch_num = 0
            predictions, references = [], []
            for batch_x, batch_y in valid:
                batch_num += 1
                #print("Validation batch ", batch_num)
                #print("len(batch_x): ", len(batch_x))
                batch_x, batch_x_m, batch_y, batch_y_m = data.prepare_data(batch_x, batch_y)
                #print("len(batch_x): ", len(batch_x))
                loss = model.test_step(batch_x, batch_y, batch_x_m, batch_y_m)
                total_val_loss += loss
                
                if compute_error_rates:
                  pred_seq, pred_scores, _ = model.predict_step(batch_x, batch_x_m, maxlen=MAX_LEN, k=BEAM_SIZE)
                  #print("len(pred_seq): ", len(pred_seq))
                  best_inds = np.argmax(pred_scores, axis=1)
                  #print("best_inds.shape: ", best_inds.shape)
                  #print("pred_scores: ", pred_scores)
                  #for p_seq in pred_seq:
                  #    for p in p_seq:
                  #        print("p: ", p)
                  #        print("decoded p: ", utility.decode_sequence(p, worddicts))
                  for b in range(len(batch_x)): #BATCHSIZE):
                      #print("Sample ", b+1)
                      p_seq = pred_seq[b][best_inds[b]]
                      #print(f"p_seq: {p_seq}, score: {pred_scores[b][best_inds[b]]}")
                      print("decoded p_seq: ", utility.decode_sequence(p_seq, worddicts))
                      #print("groundtruth_seq: ", list(batch_y[:,b]))
                      print("decoded groundtruth_seq: ", utility.decode_sequence(list(batch_y[:,b]), worddicts))
                      predictions.append(p_seq)
                  references.extend(batch_y.T)

            avg_val_loss = total_val_loss / len(valid)
            print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.4f}")

            if compute_error_rates:
                # Compute CER & WER
                #print("Validation predictions: ", predictions)
                #print("Validation references: ", references)
                cer, wer = compute_cer_wer(predictions, references, space_index)
                print(f"Validation CER: {cer:.4f}, WER: {wer:.4f}")

            # Save checkpoint
            #checkpoint_manager.save()
            #print("Checkpoint saved.\n")
            
            # Log training progress
            #with open(log_file, "a") as f:
            #    f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{model.lr.numpy():.6f}\n")
            
            # TensorBoard logging
            with summary_writer.as_default():
                tf.summary.scalar("Train Loss", avg_train_loss, step=epoch)
                tf.summary.scalar("Validation Loss", avg_val_loss, step=epoch)
                tf.summary.scalar("Learning Rate", model.lr.numpy(), step=epoch)
            
            # Save checkpoint if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_manager.save()
                print("Checkpoint saved!")
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            # Early stopping check
            if patience_counter >= PATIENCE:
                print("Early stopping triggered. Stopping training.")
                break
            
            # Learning rate decay
            if patience_counter > 0 and patience_counter % PATIENCE == 0:
                new_lr = max(model.lr.numpy() * LR_DECAY_FACTOR, MIN_LR)
                model.lr.assign(new_lr)
                print(f"Reducing learning rate to {new_lr:.6f}")
            
            print(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds.")
            
            # Log training progress
            with open(log_file, "a") as f:
                f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{model.lr.numpy():.6f}\t{time.time() - start_time:.2f} seconds.\n")

    elif args.mode == "test":
        total_test_loss = 0
        batch_num = 0
        predictions, references = [], []
        for batch_x, batch_y in test:
            batch_num += 1
            #print("Testing batch ", batch_num)
            batch_x, batch_x_m, batch_y, batch_y_m = data.prepare_data(batch_x, batch_y)
            loss = model.test_step(batch_x, batch_y, batch_x_m, batch_y_m)
            total_test_loss += loss
            
            pred_seq, pred_scores, _ = model.predict_step(batch_x, batch_x_m, maxlen=MAX_LEN, k=BEAM_SIZE)
            #print(f"pred_seq: {pred_seq}, scores: {pred_scores}")
            best_inds = np.argmax(pred_scores, axis=1)
            for b in range(len(batch_x)):
                #print("Sample ", b+1)
                p_seq = pred_seq[b][best_inds[b]]
                #print(f"p_seq: {p_seq}, score: {pred_scores[b][best_inds[b]]}")
                print("decoded p_seq: ", utility.decode_sequence(p_seq, worddicts))
                #print("groundtruth_seq: ", list(batch_y[:,b]))
                print("decoded groundtruth_seq: ", utility.decode_sequence(list(batch_y[:,b]), worddicts))
                predictions.append(p_seq)
            references.extend(batch_y.T)
            #predictions.extend(pred_seq.numpy())
            #references.extend(batch_y.numpy())
        
        avg_test_loss = total_test_loss / len(test)
        print(f"After {epoch + 1} training epochs: Testing Loss = {avg_test_loss:.4f}")

        # Compute CER & WER on test set
        cer, wer = compute_cer_wer(predictions, references, space_index)
        print(f"After {epoch + 1} training epochs: Test CER: {cer:.4f}, WER: {wer:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--dataset", type=str, choices=["PUCIT_OHUL", "KHATT"], default="PUCIT_OHUL")
    parser.add_argument("--alpha_reg", type=float, choices=[0, 1], default=1)
    args = parser.parse_args()
    
    main(args)

