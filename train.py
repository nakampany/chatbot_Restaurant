import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# intentsのパターンに含まれる各文章をループする
for intent in intents['intents']:
    tag = intent['tag']
    # タグリストに追加する
    tags.append(tag)
    for pattern in intent['patterns']:
        # 文中の各単語をトークン化する
        w = tokenize(pattern)
        # 単語リストに追加
        all_words.extend(w)
        # xyのペアに追加
        xy.append((w, tag))

# 各単語のステムとローを設定
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# 重複を除去してソート
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# 学習データの作成
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: pattern_sentence ごとの単語バッグ
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss はクラスラベルのみを必要とし、ワンホットは必要ない
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# ハイパーパラメータ 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # データセット[i]を使って i 番目のサンプルを取得できるようなインデックスをサポートする
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # len(dataset)を呼び出すとサイズがわかる
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# 損失とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# モデルの学習
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        # y が一発勝負の場合、以下のように適用する必要がある。
        # labels = torch.max(labels, 1)[1] を適用する。
        loss = criterion(outputs, labels)

        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
