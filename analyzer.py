from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta

class Analyzer:
    """
    [Calculation handler]

    """
    
    @staticmethod
    def get_metrics(predlist, labellist):
        """
        [Calculate model metrics based on model predictions and labels]

        Args:
            predlist ([torch.Tensor]): [Epoch model predictions]
            labellist ([torch.Tensor]): [Epoch labels]

        Returns:
            [float, float, float, float, np.array]: [description]
        """
        predlist, labellist = predlist.numpy(), labellist.numpy()

        accuracy = accuracy_score(labellist, predlist)
        precision = precision_score(labellist, predlist)
        recall = recall_score(labellist, predlist)
        f1 = f1_score(labellist, predlist)
        cm = confusion_matrix(labellist, predlist)

        return accuracy, precision, recall, f1, cm
    
    @staticmethod
    def split_dict(metric_dict, epoch = None):
        """
        [Split a nested dictionary into two dictionaries and add a label to all their keys
        If epoch is specified, returns two dictionaries with added label to all their keys and with specified index values]

        Args:
            metric_dict ([dictionary]): [Dictionary storing model metrics]
            epoch ([int], optional): [Number indicating specified pass of the entire dataset]. Defaults to None.

        Returns:
            [dictionary, dictionary]: [description]
        """

        if(epoch is not None):
            metric_dict = {key: {name: metric_dict[key][name][epoch] for name in metric_dict[key]} for key in metric_dict}
        keys = [key for key in metric_dict]
        train_dict = {str(keys[0]) + " " + key: value for key, value in metric_dict[keys[0]].items()}
        val_dict = {str(keys[1]) + " " + key: value for key, value in metric_dict[keys[1]].items()}

        return train_dict, val_dict

    @staticmethod
    def nice_print(splitted_dict):
        """
        Nicely print a dictionary

        Args:
            splitted_dict ([dictionary]): [Dictionary obtained from split_dict method]
        """
        for key, value in splitted_dict.items():
            print("{} {} {} {:.4f}".format("\t", key, ":", value))

    @staticmethod
    def display_confusion_matrix(History, classes, epoch = -1):
        """
        [Display confusion matrices at a specified epoch and save
        if epoch is not specified, displays latest confusion matrices]

        Args:
            History ([Object]): [Model History]
            classes ([list of strings]): [List of available labels]
            epoch (int, optional): [Number indicating specified pass of the entire dataset]. Defaults to -1.
        """
        df_train_cm = pd.DataFrame(History.confusion_matrix_history["Training Confusion Matrices"][epoch], index = classes, columns = classes)
        df_val_cm = pd.DataFrame(History.confusion_matrix_history["Validation Confusion Matrices"][epoch], index = classes, columns = classes)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (21, 9), dpi = 450)
        sns.heatmap(df_train_cm, annot = True, fmt = "d", cmap = "viridis", xticklabels = classes, yticklabels = classes, ax = ax1)
        ax1.set_title("Training")
        sns.heatmap(df_val_cm, annot = True, fmt = "d", cmap = "viridis", xticklabels = classes, yticklabels = classes, ax = ax2)
        ax2.set_title("Validation")
        plt.savefig("Confusion Matrix.png")
        plt.tight_layout()

    @staticmethod
    def plot_binary_metrics(History):
        """
        [Plot all available metrics and losses on a subplot and save obtained figure]

        Args:
            History ([Object]): [Model History]
        """
        plt.style.use("ggplot")
        fig = plt.figure(figsize = (21, 10), dpi = 450)
        x, y = History.metrics.items()
        metrics_to_display = ["Precision","Recall", "F1", "Loss", "Accuracy"]
        ax1 = plt.subplot2grid((2,6), (0,0), colspan=2)
        ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
        ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
        ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
        ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
        axes = [ax1, ax2, ax3, ax4, ax5]
        for i, key in enumerate(metrics_to_display):
            if(key in x[1] and y[1]):
                axes[i].plot(x[1][key], label = x[0])
                axes[i].plot(y[1][key], label = y[0])
                axes[i].set_title(key)
                axes[i].legend()
                plt.tight_layout()
        plt.savefig("metrics.png")

    @staticmethod
    def plot_metrics(History):
        plt.style.use("ggplot")
        fig = plt.figure(figsize = (21, 10), dpi = 450)
        x, y = History.metrics.items()
        metrics_to_display = ["Loss", "Accuracy"]
        ax1 = plt.subplot2grid((1,2), (0,0))
        ax2 = plt.subplot2grid((1,2), (0,1))
        axes = [ax1, ax2]
        for i, key in enumerate(metrics_to_display):
            if(key in x[1] and y[1]):
                axes[i].plot(x[1][key], label = x[0])
                axes[i].plot(y[1][key], label = y[0])
                axes[i].set_title(key)
                axes[i].legend()
                plt.tight_layout()
        plt.savefig("metrics.png")
class Timer():
    """
    [Timer to measure elapsed time over a full training cycle]
    """
    def start(self):
        self.start_time = timer()

    def get_time_from_seconds(self):
       seconds = self.end_time - self.start_time
       converted_time = timedelta(seconds=seconds)
       return str(converted_time) .split(".")[0]

    def end(self):
        self.end_time = timer()
        time_elapsed = self.get_time_from_seconds()
        print("{} {}".format("\nTotal Training Time:", time_elapsed))

class EpochTimer(Timer):
    """
    [Timer to measure elapsed time over an epoch]
    """
    def start(self, epoch):
        super().start()
        self.epoch = epoch

    def end(self):
        self.end_time = timer()
        time_elapsed = self.get_time_from_seconds()
        print("{} {} {} {}".format("\t\t\t\tEpoch:", self.epoch + 1, "Training Time:", time_elapsed))
    


