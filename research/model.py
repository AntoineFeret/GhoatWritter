from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def get_response(artist_name: str):
    model_name = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # To use our fine tuned model
    #model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")

    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="Songs.csv",
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained("./fine_tuned_model")

    input_text = artist_name + "'s song lyrics:"

    generated = model.generate(
        tokenizer.encode(input_text, return_tensors="pt"),
        max_length=100,
        temperature=0.7,
        num_return_sequences=1
    )

    return tokenizer.decode(generated[0], skip_special_tokens=True)

print(get_response("Taylor Swift"))
