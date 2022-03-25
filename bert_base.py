import torch

class BERTClass(torch.nn.Module):
    def __init__(self, model, n_class, n_nodes):
        super(BERTClass, self).__init__()
        self.l1 = model
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(n_nodes, n_class)

        self.projection_layer = torch.nn.Linear(n_nodes, 512)


    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler_out = self.projection_layer(pooler)

        output = self.classifier(pooler)
        return output, pooler_out