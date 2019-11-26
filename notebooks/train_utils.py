import time
import torch

from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, 
                optimizer,
                learning_rate,
                weight_decay, 
                gradient_threshold,
                epochs, 
                accumulation_steps,
                output_folder,
                train_generator):
    
    # Lists for plotting
    time_batch = []
    time_epoch = [0]
    loss_batch = []
    loss_epoch = []

    train_errors = []

    # Accumulation of values if updating gradients over multiple batches
    accuracy_accumulated = 0
    loss_accumulated = 0

    if optimizer == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print("Optimizer not recognized ... exit()")
        exit()
        
    criterion = nn.MSELoss()

    for e in range(epochs):
        print("Epoch : " + str(e + 1) + " / " + str(epochs) + " ...")
        
        model.train()
        errors_temp = []

        start = time.time()

        start_batch = time.time()
        for batch_idx, batch in enumerate(train_generator):
            # print("Batch : " + str(batch_idx) + " / " + str(len(train_generator)))
            
            X = batch[0].to(device)
            Y = batch[1].to(device)
            
            out = model(X)
            
            loss = criterion(out, Y)

            loss_accumulated += loss.item() / accumulation_steps
            loss.backward()
            
            # Gradient clipping
            if gradient_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)
                
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

                # Save loss per batch
                time_batch.append(e + (batch_idx / accumulation_steps) / (len(train_generator) / accumulation_steps))
                loss_batch.append(loss.item())

                batch_log = open(output_folder + "batch_log.txt", "a+")
                batch_log.write("[" + str(e) + " - " + str(int(batch_idx / accumulation_steps)) + "/"
                                + str(int(len(train_generator) / accumulation_steps)) +
                                "] loss : " + str(loss_accumulated))
                batch_log.write("\r\n")
                batch_log.close()
                errors_temp.append(1 - accuracy_accumulated)

                # print("Batch took : " + str(time.time() - start_batch) + "s")

                accuracy_accumulated = 0
                loss_accumulated = 0
                start_batch = time.time()
        
        # Save loss per epoch
        time_epoch.append(e + 1)
        loss_epoch.append(sum(loss_batch[e * len(train_generator): (e + 1) * len(train_generator)]) / len(train_generator))
        
        # Save model
        torch.save(model.state_dict(), str(output_folder) + "model" + str(e) + ".pt")
        
    return model, time_batch, loss_batch, time_epoch, loss_epoch
        

def eval_test_set(model, test_generator):
    with torch.no_grad():
        criterion = nn.MSELoss()
        test_loss = 0
        
        for batch_idx, batch in enumerate(test_generator):
            X = batch[0].to(device)
            Y = batch[1].to(device)

            out = model(X)

            loss = criterion(out, Y)
            test_loss += loss.item() / X.shape[0]
            
            if batch_idx % 10 == 0:
                print("Batch : " + str(batch_idx) + " / " + str(len(test_generator))) 
                print("Random point prediction :")
                print("-> True : (" + str(int(Y[0, 0].item() * 424)) + ", " + str(int(Y[0, 1].item() * 512)) + ")")
                print("-> Pred : (" + str(int(out[0, 0].item() * 424)) + ", " +  str(int(out[0, 1].item() * 512)) + ")")
                print("-> Loss : " + str(loss.item()))
                print()

        print("\r\n\r\n===== Average test loss " + str(test_loss) + "=====")

    