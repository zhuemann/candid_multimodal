import torch

class T5_classifier(torch.nn.Module):
    def __init__(self, model, n_class = 2):
        super(T5_classifier, self).__init__()
        self.language = model
        self.classifier = torch.nn.Linear(1024, 512)


        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, n_class)
        )

    def forward(self, ids, mask, token_type_ids):
        encoder_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = encoder_output.last_hidden_state
        lang_rep = pooled_sentence
        print(lang_rep.size)



        return lang_rep