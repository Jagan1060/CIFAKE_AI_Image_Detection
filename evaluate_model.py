import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

model = tf.keras.models.load_model("model/model.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_data)

predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_data.classes

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['AI Generated','Real'],
            yticklabels=['AI Generated','Real'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("\nClassification Report:\n")
print(classification_report(true_classes,
                            predicted_classes,
                            target_names=['AI Generated','Real']))