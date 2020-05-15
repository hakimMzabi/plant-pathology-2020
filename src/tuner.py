import importlib
import os

from src.cifar10 import Cifar10
from src.helper import Helper


class Tuner:
    """
    The Tuner class is here to automatically generate model generation scenarios present in the src/scenarios/ folder
    The number of epochs for each scenarios is equal to 100.
    TODO add multiple class extending from a Tuner interface to avoid too much if else statements
    TODO e.g. MlpTuner() or ResnetTuner() with hyperparameters lists as constructor params
    """
    def __init__(
            self,
            process_name=None,
            dropouts=None,
            dropout_values=None,
            optimizers=None,
            activation_functions=None,
            batch_sizes=None,
            filter_size=None,
            padding_values=None,
            kernel_sizes=None,
            h_filters_values=None,
            l_filters_values=None,
            n_neurons_values=None,
            n_resblock_values=None,
            n_layers=None
    ):
        if process_name is not None:
            self.process_name = process_name
            if process_name.lower() == "mlp":
                self.dropouts = dropouts
                self.dropout_values = dropout_values
                self.optimizers = optimizers
                self.activation_functions = activation_functions
                self.batch_sizes = batch_sizes
            elif process_name.lower() == "convnet":
                self.dropouts = dropouts
                self.dropout_values = dropout_values
                self.optimizers = optimizers
                self.activation_functions = activation_functions
                self.batch_sizes = batch_sizes
                self.filter_size = filter_size
                self.padding_values = padding_values
                self.kernel_sizes = kernel_sizes
            elif process_name.lower() == "resnet":
                self.h_filters_values = h_filters_values
                self.l_filters_values = l_filters_values
                self.dropout_values = dropout_values
                self.n_neurons_values = n_neurons_values
                self.kernel_sizes = kernel_sizes
                self.batch_sizes = batch_sizes
                self.n_resblock_values = n_resblock_values
            elif process_name.lower() == "rnn":
                self.optimizers = optimizers
                self.batch_sizes = batch_sizes
                self.n_neurons_values = n_neurons_values
                self.dropout_values = dropout_values
                self.n_layers = n_layers
        self.src_path = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def rnn_write(
            scenario_file,
            n_layers,
            optimizer,
            batch_size,
            n_neurons_v,
            dropout_v
    ):
        """
        Write RNN hyperparameters in a scenario file
        :param scenario_file:
        :param n_layers:
        :param optimizer:
        :param batch_size:
        :param n_neurons_v:
        :param dropout_v:
        :return:
        """
        scenario_file.write(f"{n_layers},{n_neurons_v},{dropout_v},{optimizer},{batch_size}\n")

    @staticmethod
    def resnet_write(
            scenario_file,
            n_resblock_v,
            dropout_v,
            h_filters_v,
            l_fiters_v,
            n_neurons_v,
            kernel_size,
            batch_size
    ):
        """
        Write ResNet hyperparameters in a scenario file
        :param scenario_file:
        :param n_resblock_v:
        :param dropout_v:
        :param h_filters_v:
        :param l_fiters_v:
        :param n_neurons_v:
        :param kernel_size:
        :param batch_size:
        :return:
        """
        scenario_file.write(
            f"{n_resblock_v},{dropout_v},{h_filters_v},{l_fiters_v},{n_neurons_v},{kernel_size},{batch_size}\n")

    def mlp_write(self, scenario_file, dropout, optimizer, activation_function, batch_size):
        """
        Write MLP hyperparameters in a scenario file
        :param scenario_file:
        :param dropout:
        :param optimizer:
        :param activation_function:
        :param batch_size:
        :return:
        """
        if dropout == "DropoutDescending":
            scenario_file.write(f"{dropout}{Helper.list_to_str_semic(self.dropout_values)},"
                                f"{optimizer},{activation_function},{batch_size}\n")
        elif dropout == "DropoutConstant":
            for dropout_value in self.dropout_values:
                scenario_file.write(
                    f"{dropout}[{(str(dropout_value) + ';') * len(self.dropout_values)}],"
                    f"{optimizer},{activation_function},{batch_size}\n")
        else:
            scenario_file.write(
                f"{dropout},{optimizer},{activation_function},{batch_size}\n")

    def convnet_write(self, scenario_file, dropout, optimizer, activation_function, batch_size, filter_size,
                      padding_value, kernel_size):
        """
        Write ConvNet hyperparameters in a scenario file
        :param scenario_file:
        :param dropout:
        :param optimizer:
        :param activation_function:
        :param batch_size:
        :param filter_size:
        :param padding_value:
        :param kernel_size:
        :return:
        """
        if dropout == "DropoutDescending":
            scenario_file.write(f"{dropout}{Helper.list_to_str_semic(self.dropout_values)},"
                                f"{optimizer},{activation_function},{batch_size},"
                                f"{filter_size},{padding_value},{kernel_size}\n")
        elif dropout == "DropoutConstant":
            for dropout_value in self.dropout_values:
                scenario_file.write(
                    f"{dropout}[{(str(dropout_value) + ';') * len(self.dropout_values)}],"
                    f"{optimizer},{activation_function},{batch_size},{filter_size},{padding_value},{kernel_size}\n")
        else:
            scenario_file.write(
                f"{dropout},{optimizer},{activation_function},{batch_size},{filter_size}"
                f",{padding_value},{kernel_size}\n")

    def inspect_scenario(self, scenario_name):
        """
        Display every line of a scenario file
        :param scenario_name:
        :return:
        """
        scenario_file_path = self.src_path + "\\scenarios\\" + self.process_name + "\\" + scenario_name + ".csv"
        try:
            scenario_file = open(scenario_file_path, "r")
            for line in scenario_file:
                print(line, end="")
        except FileNotFoundError:
            print(f"Couldn't open {scenario_file_path}")

    def create_scenario(self, scenario_name):
        """
        Create a scenario from the hyperparameters given in a Tuner() constructor
        :param scenario_name:
        :return:
        """
        Helper.create_dir(self.src_path + "\\scenarios")
        Helper.create_dir(self.src_path + "\\scenarios\\" + self.process_name)
        scenario_file_path = self.src_path + "\\scenarios\\" + self.process_name + "\\" + scenario_name + ".csv"
        try:
            scenario_file = open(scenario_file_path, "w")
            if self.process_name == "resnet":
                for n_resblock_v in self.n_resblock_values:
                    for dropout_v in self.dropout_values:
                        for h_filters_v in self.h_filters_values:
                            for l_fiters_v in self.l_filters_values:
                                for n_neurons_v in self.n_neurons_values:
                                    for kernel_size in self.kernel_sizes:
                                        for batch_size in self.batch_sizes:
                                            self.resnet_write(
                                                scenario_file=scenario_file,
                                                n_resblock_v=n_resblock_v,
                                                dropout_v=dropout_v,
                                                h_filters_v=h_filters_v,
                                                l_fiters_v=l_fiters_v,
                                                n_neurons_v=n_neurons_v,
                                                kernel_size=kernel_size,
                                                batch_size=batch_size
                                            )
            elif self.process_name == "rnn":
                for n_layer in self.n_layers:
                    for optimizer in self.optimizers:
                        for batch_size in self.batch_sizes:
                            for n_neuron_v in self.n_neurons_values:
                                for dropout_v in self.dropout_values:
                                    self.rnn_write(
                                        scenario_file,
                                        n_layer,
                                        optimizer,
                                        batch_size,
                                        n_neuron_v,
                                        dropout_v
                                    )
            else:
                for dropout in self.dropouts:
                    for optimizer in self.optimizers:
                        for activation_function in self.activation_functions:
                            for batch_size in self.batch_sizes:
                                if self.process_name == "convnet":
                                    for padding_value in self.padding_values:
                                        for kernel_size in self.kernel_sizes:
                                            self.convnet_write(scenario_file=scenario_file, dropout=dropout,
                                                               optimizer=optimizer,
                                                               activation_function=activation_function,
                                                               batch_size=batch_size, filter_size=self.filter_size,
                                                               padding_value=padding_value, kernel_size=kernel_size)
                                else:
                                    self.mlp_write(scenario_file=scenario_file, dropout=dropout, optimizer=optimizer,
                                                   activation_function=activation_function, batch_size=batch_size)
            scenario_file.close()
        except ValueError:
            print(f"Error: Could not create a file \"{scenario_file_path}\".")
        else:
            print(f"Successfully created the \"{scenario_file_path}\" file.")

    @staticmethod
    def filter_dropout(dropout_str):
        """
        Parse dropout values from a scenario file
        TODO Change it so we don't need to parse too much (remove "[" and "]")
        :param dropout_str:
        :return:
        """
        if "[" in dropout_str:
            (dropout_type, dropout_values) = dropout_str.split("[")
            dropout_values = dropout_values.replace("]", "").split(";")[:-1]
            return dropout_type, [float(i) for i in dropout_values]
        return dropout_str, None

    def launch_scenario(
            self,
            process_name,
            scenario_name,
            x_train=None,
            y_train=None,
            x_test=None,
            y_test=None,
            epochs=None,
            resume_at=None,
            test=False
    ):
        """
        Launch a scenario of trainings
        :param process_name:
        :param scenario_name:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :param epochs:
        :param resume_at:
        :param test:
        :return:
        """
        scenario_file_path = self.src_path + "\\scenarios\\" + process_name + "\\" + scenario_name + ".csv"
        scenario_file = open(scenario_file_path, "r")
        process = importlib.import_module("src.models.processes." + process_name)
        model = None

        if test:
            config_list = []

        i = 0 if resume_at is not None else None
        for line in scenario_file:
            if resume_at is not None and i <= resume_at:
                i += 1
                if i == resume_at:
                    print(f"Resuming the scenario at line {i} ({line})")
                elif i < resume_at:
                    continue
            hp = list(map(str.strip, line.split(",")))
            if process_name == "resnet":
                n_resblocks = int(hp[0])
                dropout = float(hp[1])
                h_filters = int(hp[2])
                l_filters = int(hp[3])
                n_neurons = int(hp[4])
                k_size = int(hp[5])
                batch_size = int(hp[6])
                model = process.create_model(
                    n_resblocks=n_resblocks,
                    h_filters=h_filters,
                    l_filters=l_filters,
                    dropout=dropout,
                    n_neurons=n_neurons,
                    k_size=k_size
                )
            elif process_name == "rnn":
                n_layers = int(hp[0])
                n_neurons = int(hp[1])
                dropout = float(hp[2]) if hp[2] != "None" else None
                optimizer = hp[3]
                batch_size = int(hp[4])
                model = process.create_model(
                    n_layers,
                    optimizer,
                    n_neurons,
                    dropout
                )
            elif process_name in ["convnet", "mlp"]:
                (dropout, dropout_values) = self.filter_dropout(hp[0])
                optimizer = hp[1]
                activation_function = hp[2]
                batch_size = int(hp[3])
                if process_name == "convnet":
                    filter_size = int(hp[4])
                    padding_value = hp[5]
                    ksv_1, ksv_2 = int(hp[6].strip().replace("(", "")), int(
                        hp[7].strip().replace(")",
                                              ""))
                    kernel_size = (ksv_1, ksv_2)
                    model = process.create_model(
                        optimizer=optimizer,
                        dropout_values=dropout_values,
                        activation=activation_function,
                        filters=filter_size,
                        padding_value=padding_value,
                        kernel_size=kernel_size
                    )
                elif process_name == "mlp":
                    model = process.create_model(
                        optimizer=optimizer,
                        dropout_values=dropout_values,
                        activation=activation_function,
                    )
            else:
                raise ValueError("Model tuning for this process is not possible")

            if test:
                config_list.append(model.get_config())
            else:
                model.summary()
                Helper().fit(
                    model=model,
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    process_name=process_name,
                    hp_log_title=line
                )

        scenario_file.close()
        if test:
            return config_list

    @staticmethod
    def mlp_tuner():
        """
        MLP scenario creation method
        :return:
        """
        mlp_tuner = Tuner(
            "mlp",
            dropouts=["NoDropout", "DropoutDescending", "DropoutConstant"],
            dropout_values=[0.2, 0.1],
            optimizers=["SGD", "Adam", "Adamax"],
            activation_functions=["tanh", "relu", "sigmoid"],
            batch_sizes=[32, 64, 128, 256]
        )

        # Create scenario
        mlp_tuner.create_scenario("scenario_1")

    @staticmethod
    def convnet_tuner():
        """
        ConvNet scenario creation method
        :return:
        """
        # Create tuner
        convnet_tuner = Tuner(
            "convnet",
            dropouts=["NoDropout", "DropoutDescending"],
            dropout_values=[0.5, 0.4, 0.3],
            activation_functions=["tanh", "relu", "sigmoid"],
            batch_sizes=[512, 1024],
            filter_size=64,
            padding_values=["same"],
            kernel_sizes=[
                (3, 3),
                (4, 4)
            ],
            optimizers=["SGD", "Adam", "Adamax"]
        )

        # Create scenario
        convnet_tuner.create_scenario("scenario_convnet_4")

    @staticmethod
    def mlp_scenario_launcher():
        """
        Launch a standard MLP scenario
        :return:
        """
        cifar10 = Cifar10(dim=1)
        tuner = Tuner()
        tuner.launch_scenario(
            "mlp",
            "scenario_1",
            cifar10.x_train,
            cifar10.y_train,
            cifar10.x_test,
            cifar10.y_test,
            epochs=100
        )

    @staticmethod
    def convnet_scenario_launcher():
        """
        Launch a standard ConvNet Scenario
        :return:
        """
        cifar10 = Cifar10(dim=3)
        tuner = Tuner()
        tuner.launch_scenario(
            "convnet",
            "scenario_convnet",
            cifar10.x_train,
            cifar10.y_train,
            cifar10.x_test,
            cifar10.y_test,
            epochs=2
        )

    def show_model(self, name, id):
        """
        Show every line of a model from its name and id
        TODO Replace to fuse name and id to follow a new convention
        :param name:
        :param id:
        :return:
        """
        model_log_file = f"{self.src_path}\\models\\logs\\{name}_{id}.log"
        f = open(model_log_file, "r")
        for line in f:
            print(line, end="")
        f.close()

    @staticmethod
    def resume_scenario(process_name, scenario_file_name, epoch, n_line, dim=1):
        """
        Resume the scenario at a given line in case of crash
        :param process_name:
        :param scenario_file_name:
        :param epoch:
        :param n_line:
        :param dim:
        :return:
        """
        cifar10 = Cifar10(dim=dim)
        tuner = Tuner()
        tuner.launch_scenario(
            process_name,
            scenario_file_name,
            cifar10.x_train,
            cifar10.y_train,
            cifar10.x_test,
            cifar10.y_test,
            epochs=epoch,
            resume_at=n_line
        )


if __name__ == "__main__":
    # tuner.mlp_scenario_launcher()
    # tuner.convnet_tuner()
    # tuner.convnet_scenario_launcher()
    # tuner.show_model("mlp", 6)
    # model_loaded = helper.load_model("mlp", 109)
    # model_loaded.summary()
    # tuner.resume_mlp_scenario(109)
    # tuner.convnet_tuner()

    # cifar10 = Cifar10(dim=3)
    #
    # tuner.launch_scenario(
    #     "convnet",
    #     "scenario_convnet_4",
    #     cifar10.x_train,
    #     cifar10.y_train,
    #     cifar10.x_test,
    #     cifar10.y_test,
    #     100
    # )

    # tuner = Tuner()

    # cifar10 = Cifar10(dim=3)
    # tuner.create_scenario("resnet_scenario")
    # tuner.launch_scenario(
    #     "resnet",
    #     "resnet_scenario",
    #     cifar10.x_train,
    #     cifar10.y_train,
    #     cifar10.x_test,
    #     cifar10.y_test,
    #     100
    # )

    cifar10 = Cifar10(dim=3)
    tuner = Tuner(
        process_name="mlp",
        dropouts=["NoDropout", "DropoutDescending", "DropoutAscending"],
        dropout_values=[0.5, 0.4, 0.3],
        optimizers=["SGD", "Adam", "Adamax"],
        activation_functions=["relu", "tanh"],
        batch_sizes=[32, 64, 128]
    )
    tuner.create_scenario("scenario_1")
    tuner.launch_scenario(
        "mlp",
        "scenario_1",
        cifar10.x_train,
        cifar10.y_train,
        cifar10.x_test,
        cifar10.y_test,
        epochs=100
    )
    # tuner.create_scenario("rnn_scenario")
    cifar10 = Cifar10(dim=2)
    tuner.launch_scenario(
        "rnn",
        "rnn_scenario",
        cifar10.x_train,
        cifar10.y_train,
        cifar10.x_test,
        cifar10.y_test,
        100
    )