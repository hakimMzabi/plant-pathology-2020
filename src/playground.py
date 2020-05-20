from src.helper import Helper
from src.plant_pathology import PlantPathology


def launch_linear_model():
    from src.models.processes.linear import create_model_pp_2
    helper = Helper()
    pp = PlantPathology()
    pp.plot_image(10)
    x_train, y_train = pp.train_generator.next()
    x_test, y_test = pp.valid_generator.next()
    model = create_model_pp_2()
    # model.fit_generator(
    #     generator=pp.train_generator,
    #     steps_per_epoch=pp.step_train(),
    #     validation_data=pp.valid_generator,
    #     validation_steps=pp.step_valid(),
    #     epochs=150
    # )
    helper.fit(
        model=model,
        x=pp.train_generator,
        steps_per_epoch=pp.step_train(),
        validation_data=pp.valid_generator,
        validation_steps=pp.step_valid(),
        process_name="linear",
        acc_loss="acc", # mandatory, since default is cat_acc and doesn't work with that model,
        earlystop=True, # add this to save some time
        epochs=150
    )
    # helper.fit(
    #     model, x_train, y_train, batch_size=512, epochs=100, validation_data=(x_test, y_test), process_name="linear",
    #     acc_loss="acc"
    # )


if __name__ == "__main__":
    launch_linear_model()
    # # summary
    # model = convnet.create_model_2(img_height=300, img_width=300)
    # model.summary()
    #
    # step_train = train_generator.n // train_generator.batch_size
    # step_valid = valid_generator.n // valid_generator.batch_size
    # step_test = test_generator.n // test_generator.batch_size
    #
    # print(step_train)
    # print(step_train)
    # print(step_train)
    #
    # history = model.fit_generator(generator=train_generator,
    #                               steps_per_epoch=step_train,
    #                               validation_data=valid_generator,
    #                               validation_steps=step_valid,
    #                               epochs=10)
    #
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    #
    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    #
    # # evaluating the model
    # model.evaluate_generator(generator=valid_generator,
    #                          steps=valid_generator.n // valid_generator.batch_size)
    #
    # # predict the output
    # # test_gen.reset()
    # test_generator.reset()
    # pred = model.predict_generator(test_generator,
    #                                steps=test_generator.n // test_generator.batch_size,
    #                                verbose=1,
    #                                workers=0)
    #
    # # map predictions
    # tmp_filenames = [id[11:20] for id in test_generator.filenames]
    # filenames = []
    # for file in tmp_filenames:
    #     f = file.split('.')
    #     filenames.append(f[0])
    #
    # df = pd.DataFrame({'image_id': filenames})
    # df = df.join(pd.DataFrame(data=pred, columns=train_generator.class_indices.keys()))
    #
    # df.head(10)
    # print(df['healthy'])
    #
    # df.to_csv("results.csv", index=False)
