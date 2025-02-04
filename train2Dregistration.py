import datetime
import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from torch.optim import Adam, AdamW
import os
import sys
import tqdm
import munch
import yaml 
import pandas as pd
import json
import sys
from le_morph.model_networks import CombinedRegistrationLEDA
from  icon_registration import data as icon_data
from utils.utils import *
from utils.data import *
import warnings
import logging

torch.cuda.empty_cache() 
os.environ["DISPLAY"] = ':0'
warnings.filterwarnings('ignore')
from pylab import rcParams
rcParams['font.size'] = 35
plt.ion()


def train(model, optimizer, train_loader, train_loader_2, val_loader,val_loader_2, args):
    logging.info(str(args))

    # logging for loss
    train_loss_meter = AverageValueMeter()
    sim_loss_meter = AverageValueMeter()
    val_loss_meter = AverageValueMeter()
    mse_loss_meter = AverageValueMeter()
    icon_loss_meter = AverageValueMeter()
    reg_icon_meter = AverageValueMeter()
    latent_loss_meter = AverageValueMeter()
    if model.model_type == "gradicon":
    
        model.register_net.assign_identity_map(next(iter(train_loader))[0].shape)
    model.train()
    model.to(args.device)

    best_val_loss = float('inf')
    
    args.current_level = 0

    if(args.load_dir !=None):
        print(f'Loading the best model from: {args.load_dir}')
        checkpoint = torch.load(os.path.join(args.load_dir, "best_model.pth"), map_location=args.device)
        model.load_state_dict(checkpoint['net_state_dict'])
        model.to(args.device)


    # Training log
    training_log = {'train_loss': [], 'mse_loss': [],'icon_loss': [], 'reg_icon_loss': [], 'val_loss': [],\
                     'test_loss': None, 'latent_loss': [], 'sim_loss': []}

    for epoch in range(args.num_epochs):

        if(args.step_roots and epoch%args.step_epochs == 0 and epoch>args.registration_burin):
            best_val_loss = float('inf')
            best_model = None
            args.current_level = min(args.max_levels, args.current_level + args.step_size)


        if(epoch>args.registration_burin):
            args.leda_loss_weight = 1
            
            args.similarity_weight = args.batch_size
        else: 
            args.leda_loss_weight = 0
            args.similarity_weight = args.batch_size # *10 for gradicon

        train_loss_meter.reset()
        sim_loss_meter.reset()
        icon_loss_meter.reset()
        reg_icon_meter.reset()
        mse_loss_meter.reset()
        latent_loss_meter.reset()

        model.train()
        iterations = 0 
        
        for batch in train_loader:
            image_A = batch[0].to(args.device)

            for batch in train_loader_2:
                image_B = batch[0].to(args.device)
                

                optimizer.zero_grad()
                results = model(image_A, image_B, args.current_level)

                registrtaion_loss = results['registration_loss']
                leda_results = results['leda_results']

                # registration 
                if model.model_type == "gradicon":
                    reg_regularizer = registrtaion_loss.inverse_consistency_loss
                elif model.model_type == "transmorph":
                    reg_regularizer = registrtaion_loss.regularizer_loss
            
                similarity_loss = registrtaion_loss.similarity_loss
                
                # LEDA
                mse_loss = leda_results['mse_loss']
                inverse_consistency_loss = leda_results['inverse_consistency_loss']
                latent_loss  = leda_results['latent_loss']

                # total loss
                loss = (args.lmbda*reg_regularizer + similarity_loss*args.similarity_weight) +\
                args.leda_loss_weight*(args.mse_lambda*mse_loss + args.icon_lambda*(inverse_consistency_loss) \
                            + args.latent_lambda*latent_loss)
                
                # backprop
                loss.backward()
                optimizer.step()
                
                # update meters for book-keeping
                reg_icon_meter.update((reg_regularizer.item()))
                train_loss_meter.update(loss.item())
                mse_loss_meter.update(mse_loss.item())
                latent_loss_meter.update(latent_loss.item())
                icon_loss_meter.update((inverse_consistency_loss.item()))
                sim_loss_meter.update((similarity_loss.item()))
                iterations = iterations + 1

        train_loss = train_loss_meter.avg
        mse_loss = mse_loss_meter.avg
        icon_loss = icon_loss_meter.avg
        reg_icon_loss = reg_icon_meter.avg
        latent_loss = latent_loss_meter.avg
        sim_loss = sim_loss_meter.avg
        
        training_log['train_loss'].append(train_loss)
        training_log['mse_loss'].append(mse_loss)
        training_log['icon_loss'].append(icon_loss)
        training_log['reg_icon_loss'].append(reg_icon_loss)
        training_log['latent_loss'].append(latent_loss)
        training_log['sim_loss'].append(sim_loss)
        if(epoch%args.step_interval_to_print==0):
            logging.info(f'Epoch [{epoch+1}/{iterations}], Train Loss: {train_loss}, ICON Loss: {icon_loss}, Reg Loss: {reg_icon_loss}, MSE Loss: {mse_loss}, Latent Loss: {latent_loss},  Similarity Loss: {sim_loss}')
                

        if(epoch%args.validation_frequency == 0):
            # Validation
            val_loss = evaluate(model, val_loader,val_loader_2, args)
            training_log['val_loss'].append(val_loss.item())

            logging.info(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss}, Validation Loss: {val_loss}')
            
            # Check if current model has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epochs_since_best_val = 0  # Reset patience counter
                
                torch.save({'net_state_dict': model.state_dict()}, os.path.join(args.output_path, "best_model.pth"))
            else:
                if(epoch>args.early_stop_start):
                    epochs_since_best_val += 1

            # Early stopping
            if(epoch>args.early_stop_start):
                if args.early_stop and epochs_since_best_val > args.early_stop_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

        if(epoch%args.plotting_frequency == 0):
            torch.save({'net_state_dict': model.state_dict()}, os.path.join(args.output_path, "current_model.pth"))
            visualize_results(model, val_loader, val_loader_2, args, epoch)
        

    

    # Save metrics to JSON file
    with open(os.path.join(args.output_path, 'metrics.json'), 'w') as f:
        json.dump(training_log, f)

def evaluate(model, data_loader, data_loader_2,  args):
    # exp_name = model.exp_name
    model_type = model.model_type
    model.eval()

    val_loss = []
    

    with torch.no_grad():
        for batch in data_loader:
            image_A = batch[0].to(args.device)
            # name_A = batch[1]
            for batch in data_loader_2:
                image_B = batch[0].to(args.device)
                # name_B = batch[1]
            
                results = model(image_A, image_B, args.current_level)

                registrtaion_loss = results['registration_loss']
                leda_results = results['leda_results']

                # registration 
                if model.model_type == "gradicon":
                    reg_regularizer = registrtaion_loss.inverse_consistency_loss.item()
                elif model.model_type == "transmorph":
                    reg_regularizer = registrtaion_loss.regularizer_loss.item()
                similarity_loss = registrtaion_loss.similarity_loss.item()
                
                # LEDA
                mse_loss = leda_results['mse_loss']
                inverse_consistency_loss = leda_results['inverse_consistency_loss']
                latent_loss  = leda_results['latent_loss']

                # total loss
                loss = (args.lmbda*reg_regularizer + similarity_loss) +\
                args.leda_loss_weight*(args.mse_lambda*mse_loss + args.icon_lambda*(inverse_consistency_loss) \
                            + args.latent_lambda*latent_loss)

                val_loss.append(loss.detach().cpu().numpy())
        
    return np.mean(val_loss)

def visualize_results(model, data_loader, data_loader_2, args, epoch):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            image_A = batch[0].to(args.device)
            # name_A = batch[1]
            for batch in data_loader_2:
                image_B = batch[0].to(args.device)
                # name_B = batch[1]
                
                results = model.predict(image_A, image_B, args.current_level)
                warped_image_A = model.register_net.warped_image_A
                identity_map = results['identity_map']
                leda_results = results['leda_results']

                phi_AB_vectorfield = results['phi_AB_vectorfield'] 
                phi_AB_displacement = results['phi_AB_disp']
                # Initialize root variables
                phi_AB_roots = [None] * 6
                phi_BA_roots = [None] * 6

        
                roots_AB, roots_BA = leda_results['phi_AB_roots'], leda_results['phi_BA_roots']
                # Assign roots based on available levels
                for i in range(min(args.max_levels, args.current_level)):
                    phi_AB_roots[i] = roots_AB[i]
                    phi_BA_roots[i] = roots_BA[i]

                root_configs = [
                    {"root": "2nd Root", "phi_AB_root": phi_AB_roots[0], "phi_BA_root": phi_BA_roots[0], "n": 2},
                    {"root": "4th Root", "phi_AB_root": phi_AB_roots[1], "phi_BA_root": phi_BA_roots[1], "n": 4},
                    {"root": "8th Root", "phi_AB_root": phi_AB_roots[2], "phi_BA_root": phi_BA_roots[2], "n": 8},
                    {"root": "16th Root", "phi_AB_root": phi_AB_roots[3], "phi_BA_root": phi_BA_roots[3], "n": 16},
                    {"root": "32nd Root", "phi_AB_root": phi_AB_roots[4], "phi_BA_root": phi_BA_roots[4], "n": 32},
                    {"root": "64th Root", "phi_AB_root": phi_AB_roots[5], "phi_BA_root": phi_BA_roots[5], "n": 64},
                ]

                plt.figure(figsize=(40,60))
                rows = 7
                cols = 4
                fig_idx = 4

                plt.subplot(rows, cols, 1)
                show_gray(image_A)
                plt.gca().set_title('Moving Image')

                plt.subplot(rows, cols, 2)
                show_gray(image_B)
                plt.gca().set_title('Fixed Image')

                plt.subplot(rows, cols, 3)
                plot_field(phi_AB_vectorfield, warped_image_A)
                plt.gca().set_title('GT Warped Image')

                plt.subplot(rows, cols, 4)
                show_gray(warped_image_A)
                plot_jacobian(phi_AB_displacement)
                
                # Loop through configurations
                if(epoch>args.registration_burin):
                    for idx in range(args.current_level):

                        config = root_configs[idx]
                        
                        
                        # Calculate compositions
                        root_AB_disp = config['phi_AB_root']
                        root_BA_disp = config['phi_BA_root']
                        title = config['root']
                        n = config['n']

                        root_identity_map = calculate_identity_check(root_AB_disp, root_BA_disp)

                        comp_AB_disp = calculate_composition(root_AB_disp, n)

                        phi_AB_root = root_AB_disp + identity_map
                        phi_AB_composed  = comp_AB_disp + identity_map

                        image_from_comp = get_itk_warped_image(image_A, image_B, phi_AB_composed, args, identity_map)
                        image_from_root = get_itk_warped_image(image_A, image_B, phi_AB_root, args, identity_map)


                        plt.subplot(rows, cols, fig_idx + 1)
                        plot_field(phi_AB_root, image_from_root)
                        # plt.colorbar()
                        plt.gca().set_title(title)


                        plt.subplot(rows, cols, fig_idx + 2)
                        plot_field(phi_AB_composed, image_from_comp)
                        # plt.colorbar()
                        plt.gca().set_title(f'Root composed {n} times')


                        plt.subplot(rows, cols, fig_idx + 3)
                        show_gray(image_from_comp)
                        plot_jacobian(root_AB_disp)
                        # plt.colorbar()
                        plt.gca().set_title(f'Log Det Jacobian: {title}')


                        plt.subplot(rows, cols, fig_idx + 4)
                        plot_field(root_identity_map)
                        # plt.colorbar()
                        plt.gca().set_title(f'Inverse Consistency Check: {title}')

                        fig_idx = fig_idx + cols


                    
                plt.tight_layout()
                plt.savefig(f'{args.output_path}/comp_check_{epoch}.png',bbox_inches='tight')
                plt.close()
                break


            break
            




def main():
    import argparse
    parser = argparse.ArgumentParser(description='Single pair exp config')
    parser.add_argument('--config', type =str, help='path to config file')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    # set seeds 
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_time = datetime.datetime.now().isoformat()[:19]
    exp_name = args.model_name  + '_' + print_time.replace(':', "-")
    log_dir = os.path.join(args.work_dir,args. dataset_name, exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                  logging.StreamHandler(sys.stdout)])

    args.output_path = log_dir
    

    if args.dataset_name == 'triangles':
        # split=None, data_size=128, hollow=False, samples=6000, batch_size=128
        train_loader, train_loader_2 = icon_data.get_dataset_triangles("train", data_size=160, hollow=False, samples=300, batch_size=args.batch_size)
        val_loader, val_loader_2 = icon_data.get_dataset_triangles("test", data_size=160, hollow=False, samples = 50, batch_size=1)
    elif args.dataset_name == 'torus':
        train_loader, train_loader_2, val_loader,val_loader_2, test_loader = get_torusbmp_data(args.batch_size)
    elif args.dataset_name == 'oasis2D':
        train_loader, train_loader_2, val_loader,val_loader_2, test_loader = get_oasis2D_data(args.batch_size)
        

    
    model = CombinedRegistrationLEDA(args=args)
    model.to(args.device)
    optimizer_choice = {'adam': Adam, 'adamw': AdamW}
    optimizer = optimizer_choice[args.optimizer](model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay, amsgrad = args.amsgrad)
    
    #model, optimizer, train_loader, val_loader, args
    train(model, optimizer, train_loader,train_loader_2, val_loader,val_loader_2, args)
    

    # Update yaml in log dir
    with open(os.path.join(args.output_path, f'{args.model_type}.yaml'), 'w') as f:
        yaml.dump(dict(args.__dict__), f)



if __name__ == '__main__':
    main()