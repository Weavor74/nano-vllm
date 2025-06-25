#!/usr/bin/env python3
"""
Example: Training a Language Model Completely From Scratch

This script demonstrates how to train a language model from scratch using only
your documents - no pre-trained model, custom vocabulary, custom architecture.
"""

import os
import json
import shutil
from pathlib import Path

import torch

from nanovllm.config import TrainingConfig
from nanovllm.training import Trainer, analyze_document_collection
from nanovllm.training.from_scratch import create_model_from_scratch, CustomTokenizer


def create_domain_specific_documents():
    """Create a domain-specific document collection for from-scratch training."""
    print("📚 Creating domain-specific document collection...")
    
    docs_dir = Path("./domain_docs")
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir()
    
    # Create documents in a specific domain (e.g., cooking/recipes)
    cooking_docs = {
        "basic_cooking.txt": """
Cooking Fundamentals

Cooking is the art and science of preparing food using heat. Basic cooking methods include:

Dry Heat Methods:
- Roasting: Cooking in an oven with dry heat
- Grilling: Cooking over direct heat source
- Sautéing: Cooking quickly in a small amount of fat
- Frying: Cooking in hot oil or fat

Moist Heat Methods:
- Boiling: Cooking in bubbling water at 212°F
- Steaming: Cooking with steam from boiling water
- Braising: Combination of searing and slow cooking in liquid
- Poaching: Gentle cooking in simmering liquid

Essential cooking tools include knives, cutting boards, pots, pans, and measuring tools.
Temperature control is crucial for food safety and quality.
        """.strip(),
        
        "ingredients_guide.txt": """
Essential Cooking Ingredients

Pantry Staples:
- Salt: Enhances flavors, use kosher or sea salt
- Black pepper: Freshly ground is best
- Olive oil: Extra virgin for finishing, regular for cooking
- Garlic: Fresh cloves provide the best flavor
- Onions: Yellow onions are most versatile

Herbs and Spices:
- Basil: Fresh or dried, pairs well with tomatoes
- Oregano: Essential for Italian and Mediterranean dishes
- Thyme: Earthy flavor, good with meats and vegetables
- Paprika: Adds color and mild pepper flavor
- Cumin: Warm, earthy spice for Mexican and Middle Eastern cuisine

Fresh Ingredients:
- Lemons: Acid brightens flavors
- Butter: Unsalted for cooking, salted for finishing
- Eggs: Versatile protein, binding agent
- Flour: All-purpose for most baking and thickening
- Sugar: Granulated white sugar is most common

Quality ingredients make better dishes. Buy fresh when possible.
        """.strip(),
        
        "cooking_techniques.txt": """
Advanced Cooking Techniques

Knife Skills:
- Julienne: Thin matchstick cuts
- Brunoise: Fine dice, 1/8 inch cubes
- Chiffonade: Thin ribbon cuts for herbs
- Rough chop: Irregular pieces for rustic dishes

Sauce Making:
- Roux: Equal parts flour and fat, cooked together
- Emulsification: Combining oil and water-based ingredients
- Reduction: Concentrating flavors by evaporating liquid
- Deglazing: Using liquid to lift browned bits from pan

Meat Preparation:
- Searing: High heat to create flavorful crust
- Resting: Allowing meat to redistribute juices
- Marinating: Tenderizing and flavoring with acids and enzymes
- Proper internal temperatures for food safety

Vegetable Techniques:
- Blanching: Brief boiling followed by ice bath
- Roasting: High heat to caramelize natural sugars
- Grilling: Direct heat for smoky flavor
- Pickling: Preserving in acidic solution

Practice these techniques to improve your cooking skills.
        """.strip(),
        
        "recipe_basics.txt": """
Recipe Development and Cooking Tips

Reading Recipes:
- Mise en place: Prepare all ingredients before cooking
- Follow measurements precisely, especially in baking
- Understand cooking times are guidelines, not absolutes
- Taste and adjust seasonings throughout cooking

Recipe Structure:
- Ingredient list with quantities and preparation notes
- Step-by-step instructions in logical order
- Cooking times and temperatures
- Serving size and nutritional information

Common Cooking Ratios:
- Rice to water: 1:2 ratio for most rice types
- Pasta water: 1 gallon water per pound of pasta
- Salt for pasta water: 1 tablespoon per gallon
- Oil for sautéing: 1-2 tablespoons per pan

Troubleshooting:
- Too salty: Add acid (lemon juice) or dairy
- Too spicy: Add dairy, sugar, or starch
- Too bland: Add salt, acid, or aromatics
- Overcooked vegetables: Shock in ice water

Keep detailed notes when developing your own recipes.
        """.strip()
    }
    
    # Save cooking documents
    for filename, content in cooking_docs.items():
        with open(docs_dir / filename, "w") as f:
            f.write(content)
    
    # Create JSONL with recipe data
    recipes = [
        {
            "name": "Simple Pasta",
            "ingredients": ["pasta", "olive oil", "garlic", "salt", "pepper"],
            "instructions": "Boil pasta in salted water. Heat olive oil, add minced garlic. Toss cooked pasta with garlic oil. Season with salt and pepper.",
            "cooking_time": "15 minutes",
            "difficulty": "easy"
        },
        {
            "name": "Roasted Vegetables",
            "ingredients": ["mixed vegetables", "olive oil", "salt", "herbs"],
            "instructions": "Preheat oven to 425°F. Toss vegetables with olive oil and salt. Roast for 25-30 minutes until tender. Sprinkle with fresh herbs.",
            "cooking_time": "30 minutes", 
            "difficulty": "easy"
        },
        {
            "name": "Basic Chicken Breast",
            "ingredients": ["chicken breast", "salt", "pepper", "olive oil"],
            "instructions": "Season chicken with salt and pepper. Heat oil in pan over medium-high heat. Cook chicken 6-7 minutes per side until internal temperature reaches 165°F.",
            "cooking_time": "15 minutes",
            "difficulty": "medium"
        }
    ]
    
    with open(docs_dir / "recipes.jsonl", "w") as f:
        for recipe in recipes:
            # Convert recipe to text format for training
            recipe_text = f"Recipe: {recipe['name']}\n"
            recipe_text += f"Ingredients: {', '.join(recipe['ingredients'])}\n"
            recipe_text += f"Instructions: {recipe['instructions']}\n"
            recipe_text += f"Cooking time: {recipe['cooking_time']}\n"
            recipe_text += f"Difficulty: {recipe['difficulty']}"
            
            f.write(json.dumps({"text": recipe_text}) + "\n")
    
    print(f"✅ Created domain-specific documents in {docs_dir}/")
    print(f"   Files: {[f.name for f in docs_dir.glob('*')]}")
    
    return str(docs_dir)


def demonstrate_vocabulary_building(docs_path):
    """Demonstrate custom vocabulary building from documents."""
    print("\n🔤 Building Custom Vocabulary from Documents")
    print("=" * 50)
    
    # Analyze documents first
    analysis = analyze_document_collection(docs_path)
    print(f"Document collection analysis:")
    print(f"  Total files: {analysis['total_files']}")
    print(f"  Total documents: {analysis['total_documents']}")
    print(f"  Total characters: {analysis['total_characters']:,}")
    
    # Build custom tokenizer
    tokenizer = CustomTokenizer(vocab_size=8000)  # Small vocab for demo
    vocab_stats = tokenizer.build_vocab_from_documents(docs_path)
    
    print(f"\nVocabulary Statistics:")
    print(f"  Vocabulary size: {vocab_stats['vocab_size']:,}")
    print(f"  Total tokens: {vocab_stats['total_tokens']:,}")
    print(f"  Unique tokens: {vocab_stats['unique_tokens']:,}")
    print(f"  Coverage: {vocab_stats['coverage']:.2%}")
    
    print(f"\nMost common tokens:")
    for token, freq in vocab_stats['most_common']:
        print(f"  '{token}': {freq:,}")
    
    # Test tokenization
    test_text = "Cooking pasta with garlic and olive oil is simple and delicious."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTokenization test:")
    print(f"  Original: {test_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")
    
    return tokenizer


def demonstrate_model_creation(docs_path):
    """Demonstrate creating a model from scratch."""
    print("\n🏗️ Creating Model Architecture from Scratch")
    print("=" * 50)
    
    # Create different model sizes
    model_sizes = ["tiny", "small"]
    
    for size in model_sizes:
        print(f"\nCreating {size} model...")
        
        model, tokenizer, creation_info = create_model_from_scratch(
            documents_path=docs_path,
            model_size=size,
            vocab_size=8000,  # Small vocab for demo
        )
        
        print(f"  Parameters: {creation_info['parameter_count']:,}")
        print(f"  Vocabulary: {creation_info['vocab_stats']['vocab_size']:,}")
        print(f"  Coverage: {creation_info['vocab_stats']['coverage']:.2%}")
        
        # Test forward pass
        test_input = tokenizer.encode("Cooking is")
        input_tensor = torch.tensor([test_input])
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            
        print(f"  Model output shape: {logits.shape}")
        print(f"  ✅ Model created successfully!")


def demonstrate_from_scratch_training(docs_path):
    """Demonstrate complete from-scratch training."""
    print("\n🚀 From-Scratch Training Demonstration")
    print("=" * 50)
    
    # Create training configuration
    config = TrainingConfig(
        # From-scratch settings
        train_from_scratch=True,
        model_size="tiny",  # Small for demo
        vocab_size=4000,    # Small vocab for demo
        
        # Data settings
        dataset_path=docs_path,
        output_dir="./from_scratch_demo",
        
        # Training settings
        learning_rate=1e-3,  # Higher LR for from-scratch
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_seq_length=256,  # Shorter for demo
        
        # Quick training for demo
        num_train_epochs=1,
        max_train_steps=20,
        warmup_steps=5,
        
        # Optimization
        mixed_precision="no",  # Disable for demo
        gradient_checkpointing=False,
        
        # Logging
        save_steps=10,
        logging_steps=1,
        
        seed=42,
    )
    
    print("Training configuration:")
    print(f"  Model size: {config.model_size}")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Training steps: {config.max_train_steps}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer.from_scratch(config)
    
    print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Training samples: {len(trainer.train_dataset)}")
    
    # Start training
    print("\nStarting from-scratch training...")
    try:
        results = trainer.train()
        print(f"✅ Training completed: {results}")
        
        # Test the trained model
        print("\nTesting trained model...")
        trainer.model.eval()
        
        test_prompts = ["Cooking", "Recipe for", "To prepare"]
        
        for prompt in test_prompts:
            inputs = trainer.tokenizer.encode(prompt)
            input_tensor = torch.tensor([inputs])
            
            with torch.no_grad():
                outputs = trainer.model(input_ids=input_tensor)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                
                # Get next token prediction
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = trainer.tokenizer.decode([next_token_id])
                
                print(f"  '{prompt}' -> '{next_token}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def main():
    """Run the from-scratch training demonstration."""
    print("🎯 NANO-VLLM: FROM-SCRATCH TRAINING DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how to train a language model completely from scratch")
    print("using only your documents - no pre-trained models!\n")
    
    try:
        # Step 1: Create domain-specific documents
        docs_path = create_domain_specific_documents()
        
        # Step 2: Demonstrate vocabulary building
        tokenizer = demonstrate_vocabulary_building(docs_path)
        
        # Step 3: Demonstrate model creation
        demonstrate_model_creation(docs_path)
        
        # Step 4: Demonstrate training
        success = demonstrate_from_scratch_training(docs_path)
        
        if success:
            print("\n🎉 FROM-SCRATCH TRAINING DEMONSTRATION COMPLETE!")
            print("=" * 60)
            print("✅ Successfully demonstrated:")
            print("  📚 Custom vocabulary building from your documents")
            print("  🏗️ Model architecture creation from scratch")
            print("  🚀 Complete training pipeline")
            print("  🔮 Trained model inference")
            
            print("\n📖 Usage Summary:")
            print("1. 📁 Organize your domain-specific documents")
            print("2. ⚙️ Configure for from-scratch training")
            print("3. 🚀 Run: python train.py --config from_scratch_config.json --train_from_scratch")
            print("4. 🔮 Use your custom-trained model!")
            
            print("\n🎯 Your Model Will:")
            print("  ✅ Know ONLY your domain/documents")
            print("  ✅ Use vocabulary from YOUR data")
            print("  ✅ Have architecture sized for YOUR needs")
            print("  ✅ Be completely independent and private")
            
        else:
            print("\n⚠️ Some issues detected in the demonstration")
            
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
