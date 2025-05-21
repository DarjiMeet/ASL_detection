from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAvgPool2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from hand_sign_data import train_gen, val_gen

def build_model(input_shape=(224,224,3), num_classes=36):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )

    base_model.trainable=False

    x = base_model.output
    x = GlobalAvgPool2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


if __name__ == "__main__":
    model, base_model = build_model(input_shape=(224,224,3), num_classes=36)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        "best_asl_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,  # You can increase this to 15–20 for better accuracy
        callbacks=[early_stopping, checkpoint]
    )

    model.save("asl_improve_model.h5")
