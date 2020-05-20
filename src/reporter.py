from tensorflow.keras.callbacks import Callback


class Reporter(Callback):

    def __init__(self, x=None, y=None, batch_size=None, model_name=None, log_file_path=None, hp_log_title=None,
                 steps=None,acc_loss=None):
        super().__init__()
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.epoch_iter = 0
        self.model_name = model_name
        self.log_file_path = log_file_path
        self.steps = steps
        self.acc_loss = acc_loss
        if hp_log_title:
            self.hp_log_title = hp_log_title.replace("\n", "")
        else:
            self.hp_log_title = ""

    def on_train_begin(self, logs=None):
        """
        Write the header of the current trained model log file on train begin
        :param logs:
        :return:
        """
        f = open(self.log_file_path, "a")
        f.write(f"{'=' * 5}{self.model_name}({self.hp_log_title}){'=' * 5}\n")
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        """
        Write in a the current trained model log file on epoch end
        :param epoch:
        :param logs:
        :return:
        """
        loss = None
        acc = None
        if self.x is not None and self.y is not None:
            loss, acc = self.model.evaluate(x=self.x, y=self.y, batch_size=self.batch_size)
        elif self.x is not None and self.y is None:
            loss, acc = self.model.evaluate(x=self.x, steps=self.steps)
        if logs:
            self.epoch_iter += 1
            f = open(self.log_file_path, "a")
            f.write("ep {} - "
                    "rl : {} ;"
                    "racc : {} ;"
                    "l: {} ; "
                    "acc : {} ; "
                    "vl : {} ; "
                    "vacc : {}\n"
                    .format(self.epoch_iter, loss, acc, logs['loss'], logs[self.acc_loss],
                            logs['val_loss'],
                            logs[f'val_{self.acc_loss}'])
                    )
            f.close()

    def on_train_end(self, logs=None):
        """
        Reinitialize values for the next scenario if the same Reporter object is used
        :param logs:
        :return:
        """
        self.epoch_iter = 0
