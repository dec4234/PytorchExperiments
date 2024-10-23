import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def prepare_target(target):
    boxes = []
    labels = []

    for obj in target:
        xmin, ymin, width, height = obj['bbox']
        # COCO format: [x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]
        xmax = xmin + width
        ymax = ymin + height
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj['category_id'])

    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64)
    }


def get_soccer_player_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    innerModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = innerModel.roi_heads.box_predictor.cls_score.in_features

    # Replace the head with a new one (for soccer players + background)
    innerModel.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return innerModel


# Since we're detecting 1 class + background:
model = get_soccer_player_model(num_classes=2)


transform = T.Compose([T.ToTensor()])

# Load your COCO dataset
dataset = torchvision.datasets.CocoDetection(root='soccer/train/', annFile='soccer/train/_annotations.coco.json', transform=transform)

dataloader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Move the model to GPU (if available)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

if os.path.exists('soccer_player.pth'):
    print('Loaded...')
    model.load_state_dict(torch.load('soccer_player.pth'))
    model.eval()
else:
    print('Training...')
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataloader:
            # Move images to device
            images = [image.to(device) for image in images]

            # Prepare targets and move them to device
            targets = [{k: v.to(device) for k, v in prepare_target(t).items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)

            # Sum the losses (this will be a tensor, not an int)
            total_loss = sum(loss for loss in loss_dict.values())

            # Backward pass and optimizer step
            optimizer.zero_grad()
            total_loss.backward()  # Call backward on the total loss tensor
            optimizer.step()

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item()}")

    torch.save(model.state_dict(), 'soccer_player.pth')


img = Image.open('players.png').convert("RGB")

# Transform the image (convert to tensor)
transform = T.ToTensor()
img_tensor = transform(img).unsqueeze(0)

# Move model and image to the same device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
img_tensor = img_tensor.to(device)

# Set model to evaluation mode
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

# Get the bounding boxes and labels from the predictions
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()  # Confidence scores for each box

fig, ax = plt.subplots(1)

# Show the image
ax.imshow(img)

# Loop through each bounding box and label
for i, box in enumerate(boxes):
    # If you only want to display confident detections, you can filter by score
    if scores[i] > 0.5:  # Confidence threshold
        # Unpack the coordinates of the bounding box (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = box

        # Create a rectangle patch for the bounding box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Optionally, add the label above the bounding box
        # label = f"Class: {labels[i]} | Score: {scores[i]:.2f}"
        # plt.text(xmin, ymin, label, color='white', fontsize=12,
        #          bbox=dict(facecolor='red', alpha=0.5))

# Display the plot
plt.show()




