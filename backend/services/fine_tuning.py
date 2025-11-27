"""
Fine-tune BLIP VQA model using PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
This demonstrates advanced ML skills by customizing the model for better video QA performance.
"""
import torch
from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import json
import os
from typing import List, Dict
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class VideoQADataset(Dataset):
    """Custom dataset for Video QA fine-tuning"""
    
    def __init__(
        self, 
        data_path: str, 
        processor: BlipProcessor,
        max_length: int = 512
    ):
        """
        Args:
            data_path: Path to JSON file with format:
                [{"image_path": "...", "question": "...", "answer": "..."}, ...]
            processor: BLIP processor
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {item['image_path']}: {e}")
            # Return dummy data on error
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process inputs
        encoding = self.processor(
            images=image,
            text=item['question'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Process labels (answers)
        labels = self.processor.tokenizer(
            item['answer'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        labels = labels['input_ids'].squeeze(0)
        
        # Replace padding token id's of the labels by -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        encoding['labels'] = labels
        
        return encoding


def create_lora_config() -> LoraConfig:
    """
    Create LoRA configuration for efficient fine-tuning.
    LoRA allows fine-tuning with significantly fewer parameters.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # BLIP is seq2seq
        inference_mode=False,
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    )


def prepare_model_for_finetuning(
    model_name: str = "Salesforce/blip-vqa-base",
    use_lora: bool = True
) -> tuple:
    """
    Load and prepare model for fine-tuning with LoRA.
    
    Args:
        model_name: Hugging Face model name
        use_lora: Whether to use LoRA (recommended for efficiency)
    
    Returns:
        (model, processor) tuple
    """
    # Load processor and model
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    
    if use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


def create_sample_dataset(output_path: str = "sample_vqa_data.json"):
    """
    Create a sample dataset for demonstration.
    In production, you would collect real video frame + QA pairs.
    """
    sample_data = [
        {
            "image_path": "sample_frame1.jpg",
            "question": "What is the main activity in this scene?",
            "answer": "A person is cooking in the kitchen while talking on the phone"
        },
        {
            "image_path": "sample_frame2.jpg", 
            "question": "How many people are visible?",
            "answer": "Two people are visible in the scene"
        },
        {
            "image_path": "sample_frame3.jpg",
            "question": "What objects are on the table?",
            "answer": "A laptop, coffee mug, and notebook are on the table"
        },
        # Add more examples...
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Created sample dataset at {output_path}")


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    # Implement your metrics here (BLEU, ROUGE, etc.)
    return {"accuracy": 0.0}  # Placeholder


def fine_tune_blip(
    train_data_path: str,
    val_data_path: str,
    output_dir: str = "./models/blip-vqa-finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    use_lora: bool = True
):
    """
    Fine-tune BLIP model on custom video QA data.
    
    Args:
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_lora: Use LoRA for efficient fine-tuning
    """
    logger.info("🚀 Starting BLIP fine-tuning...")
    
    # Prepare model and processor
    model, processor = prepare_model_for_finetuning(use_lora=use_lora)
    
    # Create datasets
    train_dataset = VideoQADataset(train_data_path, processor)
    val_dataset = VideoQADataset(val_data_path, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Training started...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    logger.info(f"✓ Fine-tuning complete! Model saved to {output_dir}")
    logger.info(f"Training metrics: {train_result.metrics}")
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    return model, processor


def load_finetuned_model(model_path: str):
    """Load a fine-tuned model"""
    from peft import PeftModel
    
    logger.info(f"Loading fine-tuned model from {model_path}")
    
    processor = BlipProcessor.from_pretrained(model_path)
    base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    
    return model, processor


if __name__ == "__main__":
    """
    Example usage:
    
    1. First, prepare your dataset:
       - Collect video frames with questions and answers
       - Save as JSON in the format shown in VideoQADataset
    
    2. Run fine-tuning:
       python -m backend.services.fine_tuning
    """
    
    # Example: Create sample dataset
    # create_sample_dataset("train_data.json")
    # create_sample_dataset("val_data.json")
    
    # Example: Fine-tune
    # fine_tune_blip(
    #     train_data_path="train_data.json",
    #     val_data_path="val_data.json",
    #     output_dir="./models/blip-vqa-custom",
    #     num_epochs=3,
    #     use_lora=True
    # )
    
    logger.info("""
    Fine-tuning script ready!
    
    To use:
    1. Prepare your dataset (see VideoQADataset format)
    2. Update paths in this script
    3. Run: python -m backend.services.fine_tuning
    
    Benefits of fine-tuning:
    - Improved accuracy on domain-specific video content
    - Better understanding of your specific use cases
    - Customized answer styles and formats
    - LoRA enables efficient training with minimal resources
    """)
