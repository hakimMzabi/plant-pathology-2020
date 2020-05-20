import os
import shutil
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from enum import Enum
from more_itertools import take


class Helper:
    """
    Helper class is here to help the developer debugging machine learning resources and variables.
    It also helps to manage tensorflow processes.
    """

    class LogToken(Enum):
        """
        Token to take into account in the log files
        TODO modify it to take shorter elements
        """
        LOSS = "loss"
        SPARSE_ACC = "sparse_categorical_accuracy"
        VAL_LOSS = "val_loss"
        VALL_SPARSE_ACC = "val_sparse_categorical_accuracy"
        DIGITS = 16

    def __init__(self):
        self.src_path = os.path.dirname(os.path.realpath(__file__))
        self.models_responses_folder = self.src_path + "\\models\\responses\\"
        self.checkpoint_folder = self.src_path + "\\checkpoints\\"

    def get_models_last_filename(self, process_name) -> str:
        """
        Returns the last filename of the models generated returns "No model generated." if there is no file.
        :param process_name: mlp, resnet, rnn...
        :return: the last model generated (e.g if mlp_42.h5 is the file with the highest id, it returns mlp_42.h5)
        """
        models_last_num = self.get_models_last_num(process_name)
        return self.get_model_filename(process_name, models_last_num) if models_last_num != -1 else \
            process_name + "_1"

    def get_models_last_num(self, process_name) -> int:
        """
        Return the last id of the models generated in src/models/response or -1 if there is no file
        :param process_name: mlp, resnet, rnn...
        :return: the max number of the model (e.g if mlp_42.h5 is the file with the highest id, it returns 42)
        """
        last_num_to_generate = int(self.get_models_last_num_to_generate(process_name))
        return last_num_to_generate - 1 if last_num_to_generate - 1 != 0 else -1

    def get_model_filename(self, process_name, num) -> str:
        """
        Get a model filename from a number specified
        :param process_name: mlp, resnet, rnn...
        :param num: id to select for the selected process_name
        :return: the full filename for the selected process_name and num
        """
        return self.models_responses_folder + process_name.lower() + "_" + str(num) + ".h5" \
            if os.path.isdir(self.models_responses_folder) \
            else process_name + "_1"

    def get_models_last_filename_to_generate(self, process_name) -> str:
        """
        Get the last model filename to generate in the src/models/responses folder
        :param process_name: mlp, resnet, rnn...
        :return: the filename with the highest id + 1 of the models (e.g if mlp_42.h5 is the file with the highest id,
        it returns mlp_43.h5)
        """
        return self.get_model_filename(process_name, self.get_models_last_num_to_generate(process_name))

    def get_models_last_num_to_generate(self, process_name) -> int:
        """
        Get the last number of the model to generates to handle filename incrementation
        :param process_name: mlp, resnet, rnn...
        :return: the max number of the model (e.g if mlp_42.h5 is the file with the highest id, it returns 43)
        """
        self.create_dir(self.models_responses_folder)
        max = 0
        for (dirpath, dirnames, filenames) in os.walk(self.models_responses_folder):
            for filename in filenames:
                if process_name.lower() in filename:
                    number = int(filename.split(process_name.lower() + "_")[1].split('.')[0])
                    if number > max:
                        max = number
        return max + 1

    def save_model(self, model, process_name) -> None:
        """
        Saves the model in the src/models/responses folder, automatically increments from the last model created
        :param model: tensorflow keras model
        :param process_name: mlp, resnet
        :return: None
        """
        model.save(self.get_models_last_filename_to_generate(process_name))

    @staticmethod
    def show_samples(x_train, y_train) -> None:
        """
        Show samples of an image dataset
        :param x_train: features
        :param y_train: labels
        :return: None
        """
        for i in range(10):
            plt.imshow(x_train[i])
            print(y_train[i])
            plt.show()

    @staticmethod
    def create_dir(path) -> None:
        """
        Creates a directory if it doesn't exist and print an error if it is not possible
        :param path: e.g "/random_dir/the_new_dir/
        :return: None
        """
        try:
            if not os.path.isdir(path):
                os.mkdir(path)
        except OSError:
            print(f"Couldn't create the dir : {path}")
        else:
            print(f"Successfully created the dir : {path}")

    @staticmethod
    def list_to_str_semic(list) -> str:
        """
        Convert a list to a string separated by semicolons
        :param list: e.g [0, 1, 2, 3] as a list object
        :return: [0;1;2;3] as a string
        """
        res = "["
        for el in list:
            res += str(el) + ";"
        res += "]"
        return res

    @staticmethod
    def score(acc, val_acc) -> float:
        """
        Return the score from model's accuracy and validation accuracy
        :param acc:
        :param val_acc:
        :return:
        """
        p = min(acc, val_acc)
        g = max(acc, val_acc)
        return float(10 * p * (1 + (p / g)))

    @staticmethod
    def last_line(filepath) -> str:
        """
        Get the last line of a file from its absolute or relative path
        :param filepath:
        :return:
        """
        f = open(filepath)
        lines = f.readlines()
        if len(lines) >= 2:
            x = lines[-1]
            f.close()
            return x
        f.close()
        return ""

    def get_mesures(self, el, path, model_name=None) -> tuple:
        """
        Get metrics and losses from a model
        :param el:
        :param path:
        :param model_name:
        :return:
        """
        dflt_res = "None"
        if ".log" in el and model_name is None or ".log" in el and model_name in el:
            last_line = self.last_line(path + el)
            if last_line != "":
                metrics = last_line.split(';')
                loss = metrics[0].split("-")[1].split(":")[1].strip()
                acc = metrics[1].split(':')[1].strip()
                val_loss = metrics[2].split(':')[1].strip()
                val_acc = metrics[3].split(':')[1].strip()
                return loss, acc, val_loss, val_acc
        return dflt_res, dflt_res, dflt_res, dflt_res

    def read_file(self, path):
        """
        Show every line of a file
        :param path:
        :return:
        """
        f = open(path, "r")
        for line in f:
            print(line.strip())
        f.close()

    def read_log(self, model_name):
        """
        Read log of the model
        :param model_name:
        :return:
        """
        self.read_file(self.src_path + "\\models\\logs\\" + model_name + ".log")

    def scenarios_works(self, process_name) -> bool:
        """
        Returns True if every models if a process name scenario works, False otherwise
        :param process_name:
        :return: bool
        """
        from src.tuner import Tuner
        path = self.src_path + "\\scenarios\\" + process_name + "\\"
        scenarios = os.listdir(path)
        tuner = Tuner()
        list = []

        for scenario in scenarios:
            list.append(tuner.launch_scenario(
                process_name,
                scenario_name=scenario.replace(".csv", ""),
                test=True
            ))

        for el in list:
            for subel in el:
                print(subel['name'])
            if el == "" or el is None:
                return False

        return True

    def desc(self, model_name):
        """
        Return a dict of a model from its name
        :param model_name: e.g. mlp_1
        :return: dict with score, acc, loss, val_acc, val_loss
        """
        path = self.src_path + "\\models\\logs\\"
        loss, acc, val_loss, val_acc = self.get_mesures(model_name + ".log", path, model_name.split("_")[0])
        if loss != "None" and acc != "None" and val_loss != "None" and val_acc != "None":
            return {model_name: {
                "score": self.score(float(str(acc)), float(str(val_acc))),
                "acc": ("%.2f" % (float(acc) * 100)) + "%",
                "loss": loss,
                "val_acc": ("%.2f" % (float(val_acc) * 100)) + "%",
                "val_loss": val_loss,
                "state": "overfitting" if acc > val_acc else ("underfitting" if acc < val_acc else "perfect")
            }}

    def details(self, evaluated_models):
        """
        Display the details of the evaluated models
        :param evaluated_models:
        :return: the details of the evaluated_models
        """
        path = self.src_path + "\\models\\logs\\"
        res = {}
        for k, v in evaluated_models:
            loss, acc, val_loss, val_acc = self.get_mesures(k + ".log", path, k.split("_")[0])
            res[k] = {
                "score": v,
                "acc": ("%.2f" % (float(acc) * 100)) + "%",
                "loss": loss,
                "val_acc": ("%.2f" % (float(val_acc) * 100)) + "%",
                "val_loss": val_loss,
                "state": "overfitting" if acc > val_acc else ("underfitting" if acc < val_acc else "perfect")
            }
            # print("key=" + k)
            # print("value=" + str(v))
        return res

    def evaluate_models(self, n, model_name=None) -> dict:
        """
        Evaluates the current models
        :return: the n better models
        """
        path = self.src_path + "\\models\\logs\\"
        res = {}
        try:
            els = os.listdir(path)
            for k, v in enumerate(els):
                loss, acc, val_loss, val_acc = self.get_mesures(v, path, model_name)
                if loss != "None" and acc != "None" and val_loss != "None" and val_acc != "None":
                    res[v.strip(".log")] = self.score(float(str(acc)), float(str(val_acc)))
                    # model_eval[v.strip(".log")] = {"loss": loss, "acc": acc, "val_loss": val_loss, "val_acc": acc}
        except FileNotFoundError:
            print(f"Couldn't evaluate model since there is no logs in `{path}`")
        sorted_dict = {k: v for k, v in reversed(sorted(res.items(), key=lambda item: item[1]))}
        return take(n, sorted_dict.items())

    @staticmethod
    def debug_dataset_shapes(dataset_name, dataset, terminate=False) -> None:
        """
        Show dataset shapes
        Dataset must be equal to [x_train, y_train, x_test, y_test]
        :param dataset_name: e.g "cifar10"
        :param dataset: [x_train, y_train, x_test, y_test]
        :param terminate: True if you want the program to terminates, False otherwise
        :return:
        """
        print(f"[DEBUGGER] Debugging the {dataset_name} dataset :")
        print(f"[DEBUGGER]     x_train shape : {dataset[0].shape}")
        print(f"[DEBUGGER]     y_train shape : {dataset[1].shape}")
        print(f"[DEBUGGER]     x_test shape : {dataset[2].shape}")
        print(f"[DEBUGGER]     y_test shape : {dataset[3].shape}")
        if terminate:
            exit()

    @staticmethod
    def get_cifar10_prepared(dim=1) -> (tuple, tuple):
        """
        Returns the cifar10 dataset normalized and well shaped for training as 2 tuples of 4 tensors
        :return: (tuple1 : 2 training tensors of features and labels, tuple2 : 2 validation tensors of
        features and labels)
        """
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize the data
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        # Reshape the data for training
        if dim == 1:
            x_train = x_train.reshape((50000, 32 * 32 * 3))
            x_test = x_test.reshape((10000, 32 * 32 * 3))
        elif dim == 2:
            x_train = x_train.reshape((50000, 32, 32 * 3))
            x_test = x_test.reshape((10000, 32, 32 * 3))
        elif dim == 3:
            x_train = x_train.reshape((50000, 32, 32, 3))
            x_test = x_test.reshape((10000, 32, 32, 3))
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(model_filename=None) -> object:
        """
        Returns a tensorflow keras model from the filename
        :param model_filename
        :return: a tensorflow keras model instance
        """
        import tensorflow as tf
        try:
            return tf.keras.models.load_model(model_filename)
        except FileNotFoundError:
            print("Error: Couldn't load the model. Check if the file exists.")

    def get_model(self, name, id) -> object:
        """
        Returns a tensorflow keras model from a generated
        :param name:
        :param id:
        :return:
        """
        import tensorflow as tf
        savepath = f"{self.src_path}\\models\\responses\\{name}_{id}.h5"
        return tf.keras.models.load_model(savepath)

    @staticmethod
    def create_file(path) -> None:
        """
        Create a file from path, directory must exists or file won't be created
        :param path: e.g folder1\\folder2\\file.txt
        :return: None
        """
        f = open(f"{path}", "w")
        f.close()

    def purge(self, model_name=None, ckpt=False):
        """
        Purge models and/ord checkpoints
        TODO add automatic checkpoint purge
        :param model_name:
        :param ckpt:
        :return:
        """
        if ckpt:
            try:
                shutil.rmtree(f"{self.src_path}\\models\\checkpoints\\")
            except Exception:
                pass
        if model_name is not None:
            log_model = f"{self.src_path}\\models\\logs\\{model_name}.log"
            save_model = f"{self.src_path}\\models\\responses\\{model_name}.h5"
            tensorboard_dir = f"{self.src_path}\\models\\logs\\tensorboard\\fit\\"

            try:
                os.remove(log_model)
                os.remove(save_model)
                for f in os.listdir(tensorboard_dir):
                    if model_name in f:
                        shutil.rmtree(tensorboard_dir + f + "\\")
            except Exception:
                pass

    def fit(self, model, x=None, y=None, batch_size=None, epochs=1, validation_data=None, validation_steps=None,
            process_name=None,
            hp_log_title=None, std_logs=True, earlystop=False, steps_per_epoch=None, steps_train=None, steps_valid=None,
            steps_test=None, acc_loss="cat") -> object:
        """
        Fit a model and adds a checkpoint to avoid losing data in case of failure.
        Checkpoint is also useful in case of overfitting
        :param earlystop:
        :param std_logs:
        :param hp_log_title: indicate the hyperparameters to add in fit's log file
        :param model: a tensorflow keras model
        :param x: features
        :param y: labels
        :param batch_size:
        :param epochs:
        :param validation_data: test features and test labels
        :param process_name: mlp, convnet, resnet...
        :return: history of model
        """
        import tensorflow as tf
        from src.reporter import Reporter

        self.save_model(model, process_name)

        v_acc_loss = None
        if acc_loss == "cat":
            v_acc_loss = 'categorical_accuracy'
        elif acc_loss == "sparse":
            v_acc_loss = 'sparse_categorical_accuracy'
        elif acc_loss == "acc":
            v_acc_loss = 'accuracy'

        x_test = None
        y_test = None
        if x is not None and y is not None:
            (x_test, y_test) = validation_data

        model_name = self.get_models_last_filename(process_name).split("\\")[-1].replace(".h5", "")

        log_file_dir = self.src_path + "\\models\\logs\\"
        checkpoint_file_dir = self.src_path + "\\models\\checkpoints\\"
        tensorboard_log_dir = self.src_path + "\\models\\logs\\tensorboard\\fit\\"

        self.create_dir(log_file_dir)
        self.create_dir(checkpoint_file_dir)
        self.create_dir(tensorboard_log_dir)

        log_file_path = log_file_dir + model_name + ".log"
        checkpoint_file_path = checkpoint_file_dir + model_name + ".ckpt"
        tensorboard_log_current_dir = tensorboard_log_dir + model_name + "_" + datetime.datetime.now() \
            .strftime("%Y%m%d-%H%M%S") + "\\"

        self.create_dir(tensorboard_log_current_dir)

        self.create_file(log_file_path)
        self.create_file(checkpoint_file_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=True,
            verbose=1
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_current_dir, histogram_freq=1)

        earlystop_callback = None
        if earlystop:
            earlystop_callback = tf.keras.callbacks.EarlyStopping(
                monitor=f"val_{v_acc_loss}",
                min_delta=0.0001,
                patience=1
            )

        callbacks = []
        if std_logs:
            if x is not None and y is not None:
                callbacks.append(
                    Reporter(
                        x=x,
                        y=y,
                        batch_size=batch_size,
                        model_name=model_name,
                        log_file_path=log_file_path,
                        hp_log_title=hp_log_title,
                        acc_loss=v_acc_loss
                    )
                )
            elif x is not None and y is None:
                callbacks.append(
                    Reporter(
                        x=x,
                        batch_size=batch_size,
                        model_name=model_name,
                        log_file_path=log_file_path,
                        hp_log_title=hp_log_title,
                        steps=steps_valid,
                        acc_loss=v_acc_loss
                    )
                )
            else:
                print("Error while fitting.")

        callbacks.append(cp_callback)
        callbacks.append(tensorboard_callback)
        if earlystop:
            callbacks.append(earlystop_callback)

        if x is not None and y is not None:
            history = model.fit(
                x=x,
                y=y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=callbacks
            )

        elif x is not None and y is None:
            history = model.fit(
                x=x,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                validation_steps=validation_steps,
                epochs=epochs,
                callbacks=callbacks
            )
        return history

    def get_plant_pathology_prepared(self) -> (tuple, tuple, list):
        """
        Returns the plant pathology dataset normalized and well shaped for training as 2 tuples of 4 tensors
        :return: (tuple1 : 2 training tensors of features and labels, tuple2 : 2 validation tensors of
        features and labels)
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        batch_size = 32
        img_width = 300
        img_height = 300
        seed = 42

        columns = ["healthy", "multiple_diseases", "rust", "scab"]
        df = pd.read_csv(self.src_path + "\\..\\data\\plant-pathology-2020-fgvc7\\train.csv")  # train set
        df_test = pd.read_csv(self.src_path + "\\..\\data\\plant-pathology-2020-fgvc7\\test.csv")  # test set list

        df['image_id'] = df['image_id'].astype(str) + ".jpg"
        df_test['image_id'] = df_test['image_id'].astype(str) + ".jpg"

        # data augmentation
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        # training data
        train_generator = datagen.flow_from_dataframe(
            dataframe=df[:1460],
            directory=self.src_path + "\\..\\data\\plant-pathology-2020-fgvc7\\images",
            x_col="image_id",
            y_col=columns,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            class_mode="raw",
            target_size=(img_width, img_height))
        # validation data
        valid_generator = test_datagen.flow_from_dataframe(
            dataframe=df[1460:],
            directory=self.src_path + "\\..\\data\\plant-pathology-2020-fgvc7\\images",
            x_col="image_id",
            y_col=columns,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            class_mode="raw",
            target_size=(img_width, img_height))
        # test data
        test_generator = test_datagen.flow_from_dataframe(
            dataframe=df[:],
            directory=self.src_path + "\\..\\data\\plant-pathology-2020-fgvc7\\images",
            x_col="image_id",
            batch_size=1,
            seed=seed,
            shuffle=False,
            class_mode=None,
            target_size=(img_width, img_height))

        return train_generator, valid_generator, test_generator
