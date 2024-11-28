import torch
from torch.nn.utils.rnn import pad_sequence
from mingru.mingru import MinGRU
import tqdm
import random

class MinGRUEncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int
    ):
        super(MinGRUEncoderDecoder, self).__init__()
        self.pad_token_id = pad_token_id
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.encoder = MinGRU(embedding_size, [hidden_size] * num_layers)
        self.decoder = MinGRU(embedding_size, [hidden_size] * num_layers)
        self.hidden_to_embed = torch.nn.Linear(hidden_size, embedding_size)
        self.out = torch.nn.Linear(embedding_size, vocab_size)
        self.out.weight = self.embedding.weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Shape assertions
        batch_size, input_seq_length = inputs.shape
        target_batch_size, output_seq_length = targets.shape
        if batch_size != target_batch_size:
            raise ValueError()

        input_lengths = (inputs == self.pad_token_id).float().argmax(dim=1)
        _, h = self.encoder(self.embedding(inputs), seq_lengths=input_lengths)
        top_hidden_states, _ = self.decoder(self.embedding(targets), h=h)
        preds = self.out(self.hidden_to_embed(top_hidden_states))
        return preds


def train(
    lang: str,
    num_epochs: int = 10,
    batch_size: int = 4,
    embedding_size = 64,
    hidden_size = 128,
    num_layers = 4
):
    # Load data
    dataset: dict[str, list[dict]] = {}
    for split in ["train", "dev", "test"]:
        dataset[split] = []
        with open(f"./2020/task1/data/{split}/{lang}_{split}.tsv") as f:
            for line in f:
                [input_string, output_string] = line.split("\t")
                dataset[split].append({"input": input_string, "target": output_string.split()})

    # Create vocabulary
    vocabulary = set()
    for row in dataset["train"]:
        vocabulary.update(row["input"])
        vocabulary.update(row["target"])
    vocabulary = ["<unk>", "<pad>", "<bos>", "<eos>"] + sorted(vocabulary)

    # Tokenize inputs
    tokenized_data: dict[str, list[dict]] = {}
    for split in ["train", "dev", "test"]:
        tokenized_data[split] = []
        for row in dataset[split]:
            input_ids = [vocabulary.index(c) if c in vocabulary else 0 for c in row["input"]]
            target_ids = [vocabulary.index(c) if c in vocabulary else 0 for c in row["target"]]
            target_ids = [2] + target_ids
            label_ids = target_ids + [3]
            tokenized_data[split].append({
                "input_ids": input_ids,
                "target_ids": target_ids,
                "label_ids": label_ids
            })

    # Training
    model = MinGRUEncoderDecoder(
        vocab_size=len(vocabulary),
        pad_token_id=1,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optim = torch.optim.AdamW(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    def _pad_batch(batch: list[dict]):
        input_ids = pad_sequence([torch.Tensor(row["input_ids"]) for row in batch], batch_first=True, padding_value=1).long()
        target_ids = pad_sequence([torch.Tensor(row["target_ids"]) for row in batch], batch_first=True, padding_value=1).long()
        label_ids = pad_sequence([torch.Tensor(row["target_ids"]) for row in batch], batch_first=True, padding_value=1).long()
        return input_ids, target_ids, label_ids

    num_train_batches = -(len(tokenized_data["train"]) // -batch_size)
    num_eval_batches = -(len(tokenized_data["eval"]) // -batch_size)

    pbar = tqdm.tqdm(total=num_epochs * num_train_batches, desc="Training")
    for epoch in range(num_epochs):
        model.train()
        random.shuffle(tokenized_data["train"])
        train_loss = 0
        for batch_index in range(num_batches):
            batch = tokenized_data["train"][batch_index * batch_size:(batch_index + 1) * batch_size]
            input_ids, target_ids, label_ids = _pad_batch(batch)
            preds: torch.Tensor = model(inputs=input_ids, targets=target_ids) # (B, S, V) logits
            batch_loss = loss_fn(preds.permute(0, 2, 1), label_ids)

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            train_loss += batch_loss.detach().item()
            pbar.update(1)

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch_index in range(num_eval_batches):
                batch = tokenized_data["dev"][batch_index * batch_size:(batch_index + 1) * batch_size]
                input_ids, target_ids, label_ids = _pad_batch(batch)
                preds: torch.Tensor = model(inputs=input_ids, targets=target_ids)
                batch_loss = loss_fn(preds.permute(0, 2, 1), label_ids)
                eval_loss += batch_loss.detach().item()

        avg_train_loss = train_loss / num_train_batches
        avg_eval_loss = eval_loss / num_eval_batches
        print(f"Epoch {epoch}: Train loss={avg_train_loss}\tEval loss={avg_eval_loss}")

    pbar.close()

if __name__ == "__main__":
    train("ady")
