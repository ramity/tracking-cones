import core

core.train(
    train_dir="/data/renders",
    val_dir="/data/renders",
    epochs=100,
    batch_size=69,
    lr=0.0001,
    save_path="/data/fc_model_69_0.00025_no_dist_5.pkl",
    load_path="/data/fc_model_69_0.00025_no_dist_4.pkl",
)
