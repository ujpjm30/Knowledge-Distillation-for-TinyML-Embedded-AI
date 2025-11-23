import pandas as pd
import numpy as np
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio

import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random

import warnings
warnings.filterwarnings('ignore')

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"] 
LABEL_MAP = {label: idx for idx, label in enumerate(LABELS)}
INVERSE_LABEL_MAP = {idx: label for label, idx in LABEL_MAP.items()}
print(LABEL_MAP)

PROCESSOR = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
# MODEL = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', num_labels=7)
DATA_DIR = './data/TESS Toronto emotional speech set data'



class TESSDataset(Dataset):
    def __init__(
        self,
        processor=PROCESSOR, 
        data_dir=None,  
        max_length=32000, 
        df=None,
        ):

        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length

        if df is not None:
            self.df = df
        elif data_dir is not None:
            self.df = self._load_data()
        else:
            raise ValueError("Either 'df' or 'data_dir' must be provided.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_paths']
        label = self.df.iloc[idx]['labels']

        #load audio file
        speech, sr = librosa.load(audio_path, sr=16000)

        #pad or truncate the speech to the required length
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')

        #preprocess the audio file
        inputs = self.processor(speech, sampling_rate=16000, return_tensors='pt', padding=True, truncate=True, max_length=self.max_length)

        input_values = inputs.input_values.squeeze()
        return{'input_values': input_values, 'labels': torch.tensor(label, dtype=torch.long)}

    def _load_data(self):
        data_paths = []
        labels = []
        df = pd.DataFrame()

        for dirname, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith('.wav'):
                    data_paths.append(os.path.join(dirname, filename))
                    label = filename.split('_')[-1]
                    label = label.split('.')[0]
                    #print(label)
                    labels.append(label.lower())
                if len(data_paths) == 2800:
                    break
        print('Dataset is loaded')

        df['audio_paths'] = data_paths
        df['labels'] = labels
        df['labels'] = df['labels'].map(LABEL_MAP)
        df['labels'] = df['labels'].astype(int)
        
        return df

    def split_data(self, test_size=0.2, random_state=42):
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_df, test_df

# Global variables to hold data set split
train_dataset = None
test_dataset = None

def split_datasets():
    global train_dataset, test_dataset
    dataset = TESSDataset(processor=PROCESSOR, data_dir=DATA_DIR)
    
    print(f"dataset size: {len(dataset)}")
    train_df, test_df = dataset.split_data()

    train_dataset = TESSDataset(processor=PROCESSOR, df=train_df)
    test_dataset = TESSDataset(processor=PROCESSOR, df=test_df)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(f"Train_dataset first item input value: {train_dataset[0]}")
    print(f"Test_dataset first item input value: {test_dataset[0]}")

    print(f"Train_dataset first item input value size: {train_dataset[0]['input_values'].size()}")
    print(f"Test_dataset first item input value size: {test_dataset[0]['input_values'].size()}")

    output_dir = '.data/TESS Toronto emotional speech set data/saved_splits/'
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_dataset, os.path.join(output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_dir, 'test_dataset.pt'))

def clean_directories():
    directories_to_clean = ['./teacher_results', './teacher_metrics']
    for dir_path in directories_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)

def train_model():
    clean_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        'facebook/wav2vec2-base', 
        num_labels=7,
        cache_dir=None,
        local_files_only=False
        ).to(device)

    # data_dir = './data/TESS Toronto emotional speech set data'  
    # dataset = TESSDataset(processor=PROCESSOR, data_dir=data_dir)
    # print(f"dataset size: {len(dataset)}")
    # train_df, test_df = dataset.split_data()

    # train_dataset = TESSDataset(processor=PROCESSOR, df=train_df)
    # test_dataset = TESSDataset(processor=PROCESSOR, df=test_df)

    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")

    # print(f"Train_dataset first item input value: {train_dataset[0]}")
    # print(f"Test_dataset first item input value: {test_dataset[0]}")

    # print(f"Train_dataset first item input value size: {train_dataset[0]['input_values'].size()}")
    # print(f"Test_dataset first item input value size: {test_dataset[0]['input_values'].size()}")
    total_params = sum(p.numel() for p in model.parameters())
    mean_value = sum(p.mean().item() for p in model.parameters()) / len(list(model.parameters()))
    print(f"\nInitial model state:")
    print(f"Number of parameters: {total_params}")
    print(f"Mean parameter value: {mean_value}")

    training_args = TrainingArguments(
        output_dir='./teacher_results',
        overwrite_output_dir=True,
        load_best_model_at_end=False,
        resume_from_checkpoint=False,
        evaluation_strategy='steps',
        eval_steps=100,
        logging_strategy='steps',
        logging_steps=100,
        save_strategy='epoch',
        learning_rate=14e-7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.0065,
        #lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        fp16=True,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model('./teacher_results')
    
    results = trainer.evaluate()
    print(results)

    metrics_df = generate_training_metrics(trainer, output_dir='./teacher_metrics')
    print("Training metrics generated and saved")

    confusion_mat = generate_confusion_matrix(trainer, output_dir='./teacher_metrics')
    print("Confusion matrix generated and saved")

    return trainer

def eval_model(trainer):
    idx = random.randrange(0, len(trainer.eval_dataset))
    print("Original Label:", INVERSE_LABEL_MAP[int(trainer.eval_dataset[idx]['labels'])])
    input_values = trainer.eval_dataset[idx]['input_values'].unsqueeze(0).to('cuda')

    with torch.no_grad():
        outputs = trainer.model(input_values)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    print('Predicted Label: ', INVERSE_LABEL_MAP[predicted_class])

    
def compute_metrics(pred):
    labels = pred.label_ids # original labels
    preds = np.argmax(pred.predictions, axis=1) # model predicted labels
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return{
        "accuracy":accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
def test():
    data_dir = './data/TESS Toronto emotional speech set data'
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    #print(data_dir)

    dataset = TESSDataset( processor=processor, data_dir=data_dir)
    dataset.load_data()
    # Test __getitem__ for specific indices
    print(f"First item: {dataset[0]}")  # Should return a dictionary with 'input_values' and 'labels'
    print(f"Last item: {dataset[len(dataset) - 1]}")  # Should return the last item in the dataset
    
    item = dataset[0]
    print(f"Input values shape: {item['input_values'].shape}")  # Verify dimensions
    print(f"Label: {item['labels']}")  # Verify label is correct

    for i in range(len(dataset)):
        item = dataset[i]
        if i < 5:  # Limit the output to first few for clarity
            print(f"Item {i}: {item}")


def generate_training_metrics(trainer, output_dir='./metrics'):
    history = trainer.state.log_history
    
    # Create metrics dictionary
    metrics_dict = {
        'step': [],
        'accuracy': [],
        'precision': [], 
        'recall': [],
        'f1': [],
        'loss': []  # Adding loss tracking
    }
    
    train_metrics = []
    eval_metrics = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            train_metrics.append({
                'step': entry['step'],
                'loss': entry['loss']
            })
        elif 'eval_accuracy' in entry:
            eval_metrics.append({
                'step': entry['step'],
                'accuracy': entry['eval_accuracy'],
                'precision': entry['eval_precision'],
                'recall': entry['eval_recall'],
                'f1': entry['eval_f1'],
                'loss': entry.get('eval_loss', None)
            })
    
    # Create separate DataFrames
    train_df = pd.DataFrame(train_metrics)
    eval_df = pd.DataFrame(eval_metrics)
    
    # Save separate CSV files
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'training_loss.csv'), index=False)
    eval_df.to_csv(os.path.join(output_dir, 'eval_metrics.csv'), index=False)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['step'], train_df['loss'], label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    # Plot evaluation metrics
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in eval_df.columns:
            plt.plot(eval_df['step'], eval_df[metric], 
                    label=metric.capitalize(), marker='o')
    
    plt.xlabel('Training Step')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics Over Steps')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'eval_metrics.png'))
    plt.close()
    
    return eval_df

def generate_confusion_matrix(trainer, output_dir='./metrics'):
    predictions = trainer.predict(trainer.eval_dataset)
    
    # Convert logits to predicted class indices
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix with seaborn for better styling
    sns.heatmap(cm, 
                annot=True,  # Show numbers in cells
                fmt='d',     # Use integer format
                cmap='Blues',
                xticklabels=LABELS,  # Using your existing LABELS 
                yticklabels=LABELS)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    return cm

if __name__ == "__main__":
    split_datasets()
    trained_model = train_model()
    # eval_model(trained_model)
    #test()