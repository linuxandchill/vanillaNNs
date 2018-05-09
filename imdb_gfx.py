from keras.models import load_model 

model = load_model('imdb_val.h5')

#history_dict = history.history

model.fit(epochs=20)
