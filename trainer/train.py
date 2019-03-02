import argparse
import torch.optim as optim
from torchvision import datasets, transforms

from model import CNN_small
from dataset import RoadSceneDataset


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_args():
    parser = argparse.ArgumentParser(description='Training model for the synthetic road scene')
    # Training settings
    parser.add_argument('--data', type=string, help='Path to training and evaluation data')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--eval-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='max number of epochs to train (default: 10)')
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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    return parser.parse_args()

def load_data(data_dir):
    # Load training and evaluation data
    # TODO
    train_data['images']
    train_data['offsets']
    train_data['angles']
    eval_data['images']
    eval_data['offsets']
    eval_data['angles']
    return train_data, eval_data

def main():
    args = get_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data, eval_data = load_data(args.data)
    train_dataset = RoadSceneDataset(train_data['images'], train_data['offsets'], train_data['angles'])
    eval_dataset = RoadSceneDataset(eval_data['images'], eval_data['offsets'], eval_data['angles'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size, shuffle=False)

    model = CNN_small().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TODO, starts here
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.set_trace()
