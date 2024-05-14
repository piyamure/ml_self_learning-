# pytorch libs 
import torch
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])

# Create DataLoaders with help from data_setup.py
import data_setup
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE)

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
import model_builder
# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

# Start training with help from engine.py
import engine
engine.train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,
    loss_fn=loss_fn,optimizer=optimizer,epochs=NUM_EPOCHS,device=device)

# Save the model with help from utils.py
import utils
utils.save_model(model=model,target_dir="models", model_name="05_going_modular_script_mode_tinyvgg_model.pth")