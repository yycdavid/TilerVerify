import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import os
import pickle
import utils
from PIL import Image
import torch.nn.functional as F
import scipy.io as sio

from model import CNN_small
from dataset import RoadSceneDataset

outputManager = 0
RESULTS_ROOT = 'trained_models'
MAX_NON_IMPROVING_EPOCHS = 10

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (images, offsets, angles) in enumerate(train_loader):
        images, offsets, angles = images.to(device), offsets.to(device), angles.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = F.l1_loss(output, torch.cat([offsets.unsqueeze(1), angles.unsqueeze(1)], dim=1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            outputManager.say('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(offsets), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, offsets, angles in test_loader:
            images, offsets, angles = images.to(device), offsets.to(device), angles.to(device)
            output = model(images)
            test_loss += F.l1_loss(output, torch.cat([offsets.unsqueeze(1), angles.unsqueeze(1)], dim=1), reduction='sum').item() # sum up batch loss

    test_loss /= (2*len(test_loader.dataset))
    outputManager.say('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss

def get_args():
    parser = argparse.ArgumentParser(description='Training model for the synthetic road scene')
    # Training settings
    parser.add_argument('--train_data', type=str, help='File of training data')
    parser.add_argument('--val_data', type=str, help='File of validation data')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='max number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--result', type=str, help='Folder name to store result')

    return parser.parse_args()

def load_data(data_dir, train_file, val_file):
    # Load training and evaluation data
    train_data = sio.loadmat(os.path.join(data_dir, train_file))
    eval_data = sio.loadmat(os.path.join(data_dir, val_file))
    return train_data, eval_data

def main():
    args = get_args()
    assert args.result is not None, 'Need to specify result directory'
    # Create folder to store results for this experiment
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    result_root = os.path.join(base_dir, RESULTS_ROOT)
    if not os.path.exists(result_root):
        print("Creating {}".format(result_root))
        os.makedirs(result_root)
    result_dir = os.path.join(result_root, args.result)
    utils.create_dir(result_dir)
    global outputManager
    outputManager = utils.OutputManager(result_dir)

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = os.path.join(base_dir, 'data')
    train_data, eval_data = load_data(data_dir, args.train_data, args.val_data)

    train_dataset = RoadSceneDataset(train_data['images'], train_data['offsets'].squeeze(), train_data['angles'].squeeze())
    eval_dataset = RoadSceneDataset(eval_data['images'], eval_data['offsets'].squeeze(), eval_data['angles'].squeeze())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size, shuffle=False)

    model = CNN_small().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # log configuration
    with open(os.path.join(result_dir, 'config.txt'), 'w') as f:
        f.write(model.__repr__())
        f.write('Input dimension: {}'.format(train_data['images'][0].shape))

    best_epoch = 0
    best_valid_loss = None
    for epoch in range(1, args.epochs + 1):
        if (epoch - best_epoch) > MAX_NON_IMPROVING_EPOCHS:
            break
        train(args, model, device, train_loader, optimizer, epoch)
        eval_loss = test(args, model, device, eval_loader)
        if best_valid_loss is None or eval_loss < best_valid_loss:
            outputManager.say(
                "Found better model ({} < {}) @ epoch {}".format(
                    eval_loss,
                    best_valid_loss,
                    epoch,
                )
            )
            best_valid_loss = eval_loss
            best_epoch = epoch
            outputManager.say('Saving the current model...')
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pt"))

    outputManager.say('Training finished. The best model is @ epoch {}'.format(best_epoch))

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
