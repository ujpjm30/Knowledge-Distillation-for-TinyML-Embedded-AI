from transformers import TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from teacher_model_wav2vec2base import LABELS, LABEL_MAP, INVERSE_LABEL_MAP, DATA_DIR, TESSDataset, generate_training_metrics, generate_confusion_matrix
from transformers import Wav2Vec2ForSequenceClassification
import os
import torch
import numpy as np
from evaluate import load
import student_model_wav2small
from typing import Optional, Tuple
import json
from safetensors import safe_open
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

accuracy_metric = load("accuracy")

def load_data_splits():
    # Define the directory and file paths
    output_dir = '.data/TESS Toronto emotional speech set data/saved_splits/'  # Same directory as used for saving
    train_dataset_path = os.path.join(output_dir, 'train_dataset.pt')
    test_dataset_path = os.path.join(output_dir, 'test_dataset.pt')

    # Load the datasets
    train_dataset = torch.load(train_dataset_path)
    test_dataset = torch.load(test_dataset_path)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def initialize_student_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model_wav2small.Wav2Small().to(device)

    # print(f"\nStudent model initialized on {device}")
    # print(f"Number of parameters: {sum(p.numel() for p in student_model.parameters())}")
    
    # Verify this is a fresh model by checking parameter statistics
    total_params = sum(p.numel() for p in student_model.parameters())
    mean_value = sum(p.mean().item() for p in student_model.parameters()) / len(list(student_model.parameters()))
    
    print(f"\nStudent model initialized on {device}")
    print(f"Number of parameters: {total_params}")
    print(f"Mean parameter value: {mean_value}")  # This should be close to 0 for a fresh model
    
    return student_model

class KnowledgeDistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

# New loss function / Knowledge Distillation loss function: overriding the compute_loss() method to include the knowledge distillation loss term.
class KnowledgeDistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract cross entropy loss and logits from student model
        outputs_student = model (**inputs)
        loss_ce = outputs_student.loss
        logits_student = outputs_student.logits

        #extract logits from teacher
        outputs_teacher = self.teacher_model(**inputs)
        logits_teacher = outputs_teacher.logits

        # Compute Distillation Loss by Softening Probabilities
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        # Average the losses over the batch dimension

        loss_kd = self.args.temperature**2 * loss_fct(
            # Soften the proababilites of the student.
            F.log_softmax(logits_student/self.args.temperature, dim=1),
            # Soften the probabilities of the teacher.
            F.softmax(logits_teacher/self.args.temperature, dim=1)
        )

        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha)*loss_kd

        return (loss, outputs_student) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)

        if state_dict:
            torch.save(state_dict, os.path.join(output_dir, "trainer_state.bin"))

def compute_metrics(pred):
    predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions 
    #predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=pred.label_ids)  #references=labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        pred.label_ids,
        predictions,
        average='weighted',
        zero_division=0
    )

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_distillation():
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset = load_data_splits()

    teacher_checkpoint = "./teacher_results"
    teacher_model = Wav2Vec2ForSequenceClassification.from_pretrained(teacher_checkpoint).to(device)
    teacher_model.eval()

    student_model = initialize_student_model()

    #student_model = Wav2Small().to(device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_student_ckpt = f"./wav2small_distilled_{timestamp}"
    student_training_args = KnowledgeDistillationTrainingArguments(
        output_dir = finetuned_student_ckpt,
        overwrite_output_dir=True,
        load_best_model_at_end=False,
        save_strategy="no",
        evaluation_strategy = "epoch",
        num_train_epochs = 10,
        learning_rate = 2e-4,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        weight_decay = 0.007,
        alpha=0.5,
        temperature=2.0,
        report_to=[]
    )

    student_trainer = KnowledgeDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=student_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    student_trainer.train()

    metrics_df = generate_training_metrics(student_trainer, output_dir='./student_distillation_metrics')
    print("Training metrics generated and saved")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    confusion_mat = generate_confusion_matrix(student_trainer, output_dir='./student_distillation_metrics')
    print("Confusion matrix generated and saved")

    final_output_dir = "./wav2small_distilled_final_{timestamp}"
    student_trainer.save_model(final_output_dir)
    
    return student_trainer, test_dataset

def compute_final_metrics(trainer, test_dataset):
    """
    Compute and display final model metrics
    """
    # Get predictions on test set
    predictions = trainer.predict(test_dataset)
    
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    print("\nFinal Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def compute_parameters(model_path):
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        
        with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt", device="cpu") as f:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            parameters = model.num_parameters()
    
    elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        model = student_model_wav2small.Wav2Small()
        model.load_state_dict(state_dict)
        parameters = sum(p.numel() for p in model.parameters())
    
    else:
        raise FileNotFoundError(f"No model file found in {model_path}")
    
    return parameters

def compare_model_sizes():
    teacher_params = compute_parameters("./teacher_results")
    student_params = compute_parameters("./wav2small_distilled_final")

    print(f"Teacher Model Parameters: {teacher_params:,}")
    print(f"Student Model Parameters: {student_params:,}")
    print(f"Size Reduction: {((teacher_params - student_params)/teacher_params)*100:.2f}%")

if __name__ == "__main__":
    student_trainer, test_dataset = train_distillation()
    compute_final_metrics(student_trainer, test_dataset)
    #compare_model_sizes()
