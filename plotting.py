import matplotlib.pyplot as plt
from IPython.display import clear_output
import threading

class loss_plot:
    def __init__(self):
        self.loss_tr = []
        self.loss_te = []
        self.lr = None


    def plot(self,loss_tr,loss_te,lr):
        self.loss_tr.append(loss_tr)
        if loss_te is not None:
            self.loss_te.append(loss_te)
        self.lr = lr
        # plot train and test loss
        clear_output(wait=True)
        _, ax = plt.subplots()
        ax.plot(self.loss_tr, label="Train Loss")
        if loss_te is not None:
            ax.plot(self.loss_te, label="Test Loss")

        # live learning rate
        ax.text(
            0.95,
            0.95,
            f"LR: {self.lr:.5f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.3),
        )

        # live test loss
        if loss_te is not None:
            ax.text(
                0.95,
                0.8,
                f"Test Loss: {self.loss_te[-1]:.5f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.3),
            )
        ax.legend()
        plt.show()