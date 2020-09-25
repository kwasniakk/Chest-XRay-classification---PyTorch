import torch
import pandas as pd
from analyzer import Analyzer

class History:
    """
    [Model history created to store metrics, losses and confusion matrices over full model training]
    """
    def __init__(self, type, train_loader_len, val_loader_len):
        """
        [Initialize model history to store metrics, losses and confusion matrices over full training cycle]

        Args:
            type ([string]): [History type depending on the classification task, either "Binary" or "NonBinary]
            train_loader_len ([int]): [Number of batches in training data loader]
            val_loader_len ([int]): [Number of batches in training data loader]
        """
        self.train_loader_len = train_loader_len
        self.val_loader_len = val_loader_len
        self.type = type
        if(self.type == "NonBinary"):
            self.metrics = {
                    "Training": {
                        "Training Accuracy": []},

                    "Validation": {
                        "Validation Accuracy": []}
                        }
                        
        elif(type == "Binary"):
            self.metrics = {
                "Training": {
                        "Loss": [],
                        "Accuracy": [],
                        "Precision": [],
                        "Recall": [],
                        "F1": []},

                "Validation": {
                        "Loss": [],
                        "Accuracy": [],
                        "Precision": [],
                        "Recall": [],
                        "F1": []}
                        }


        self.confusion_matrix_history = {
                "Training Confusion Matrices": [],
                "Validation Confusion Matrices": []
                }
            

    def export(self):
        """
        [Export losses and metrics over training history]
        """
        train_dict, val_dict = Analyzer.split_dict(self.metrics)
        concat_dict = {**train_dict, **val_dict}
        export_df = pd.DataFrame.from_dict(concat_dict)
        export_df.to_excel("epoch_metrics.xlsx", float_format = "%.4f")
                                                                                
    def display(self, epoch):
       """
        [Display model metrics at a specified epoch]

       Args:
            epoch ([int]): [Number indicating specified pass of the entire dataset]
       """

       current_epoch_train_metrics, current_epoch_val_metrics = Analyzer.split_dict(self.metrics, epoch)
       Analyzer.nice_print(current_epoch_train_metrics)
       print("\n")
       Analyzer.nice_print(current_epoch_val_metrics)





class Metric:
    """
    [Class for updating and saving calculated metrics to model History]
    """
 
    def reset(self):
        """
        [Reset model predictions, labels and loss at the beginning of each epoch]
        """
        self.training_loss = 0
        self.train_predlist = torch.zeros(0, dtype = torch.long)
        self.train_labellist = torch.zeros(0, dtype = torch.long)

        self.validation_loss = 0
        self.val_predlist = torch.zeros(0, dtype = torch.long)
        self.val_labellist = torch.zeros(0, dtype = torch.long)

    
    def update(self, loss, preds, labels, phase = "Training"):
        """
        [Update loss, predictions and its corresponding labels depending on model phase]

        Args:
            loss ([torch.Tensor]): [Calculated loss for each batch]
            preds ([torch.Tensor]): [Batch model predictions]
            labels ([torch.Tensor]): [Batch Labels]
            phase ([torch.Tensor]): [Current model phase, either "Training" or "Validation"]. Defaults to "Training".
        """
        if(phase == "Training"):
            self.training_loss += loss.item() 
            self.train_predlist = torch.cat([self.train_predlist, preds.view(-1).cpu()])
            self.train_labellist = torch.cat([self.train_labellist, labels.view(-1).cpu()])
        if(phase == "Validation"):
            self.validation_loss += loss.item()
            self.val_predlist = torch.cat([self.val_predlist, preds.view(-1).cpu()])
            self.val_labellist = torch.cat([self.val_labellist, labels.view(-1).cpu()])

    def save_to_history(self, History, phase):
        """
        [Save calculated metrics and loss to a History object]

        Args:
            History ([Object]): [Model History object ]
            phase ([string]): [Current model phase, either "Training" or "Validation"]
        """

        if(phase == "Training"):
            acc, prec, rec, f1, cm = Analyzer.get_metrics(self.train_predlist, self.train_labellist)
            History.metrics[phase]["Loss"].append(self.training_loss / History.train_loader_len)
            History.confusion_matrix_history["Training Confusion Matrices"].append(cm)

            if(History.type == "Binary"):
                History.metrics[phase]["Accuracy"].append(acc)
                History.metrics[phase]["Precision"].append(prec)
                History.metrics[phase]["Recall"].append(rec)
                History.metrics[phase]["F1"].append(f1)
            
            elif(History.type == "NonBinary"):
                History.metrics[phase]["Accuracy"].append(acc)


        elif(phase == "Validation"):
            acc, prec, rec, f1, cm = Analyzer.get_metrics(self.val_predlist, self.val_labellist)
            History.metrics[phase]["Loss"].append(self.validation_loss / History.val_loader_len)
            History.confusion_matrix_history["Validation Confusion Matrices"].append(cm)

            if(History.type == "Binary"):
                History.metrics[phase]["Accuracy"].append(acc)
                History.metrics[phase]["Precision"].append(prec)
                History.metrics[phase]["Recall"].append(rec)
                History.metrics[phase]["F1"].append(f1)
            
            elif((History.type == "NonBinary")):
                History.metrics[phase]["Accuracy"].append(acc)

