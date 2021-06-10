import model_construction
import model_data_generator


fit = model_construction.model_constructtion().fit_generator(model_data_generator.model_data_gen()[0],
                              epochs=10,
                              validation_data=model_data_generator.model_data_gen()[1],
                              callbacks=[model_data_generator.model_data_gen()[2]])