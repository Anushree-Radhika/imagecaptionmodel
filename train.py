from common_imports import *
from config import model_config, train_config
from model import VisionGPT2Model, tokenizer
from dataset import get_dataloaders


class Trainer:
    def __init__(self, model_config, train_config, dls):
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        print(f"trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}")

        self.tokenizer = tokenizer

        self.scaler = GradScaler()

        self.train_dl, self.val_dl = dls
        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.0)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps,
        )

        self.metrics = pd.DataFrame()
        self.metrics[["train_loss", "train_perplexity", "val_loss", "val_perplexity"]] = None

        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2(),
        ])

    def save_model(self):
        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()
        torch.save(sd, self.train_config.model_path / "captioner.pt")

    def load_best_model(self):
        sd = torch.load(self.train_config.model_path / "captioner.pt")
        self.model.load_state_dict(sd)

    def train_one_epoch(self, epoch):
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.0

        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()
                prog.set_description(f"train loss: {loss.item():.3f}")

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)
        self.metrics.loc[epoch, ["train_loss", "train_perplexity"]] = (train_loss, train_pxp)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        prog = tqdm(self.val_dl, total=len(self.val_dl))
        running_loss = 0.0

        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()
                prog.set_description(f"valid loss: {loss.item():.3f}")

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)
        self.metrics.loc[epoch, ["val_loss", "val_perplexity"]] = (val_loss, val_pxp)
        return val_pxp

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print("unfreezing GPT2 entirely...")

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()
            prog.set_description("training")
            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            prog.set_description("validating")
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print("saving best model...")
                self.save_model()

        return {"best_perplexity": best_pxp, "best_epoch": best_epoch}

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):
        self.model.eval()

        image = Image.open(image).convert("RGB")
        image = np.array(image)
        image = self.gen_tfms(image=image)["image"]
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image, sequence,
            max_tokens=max_tokens, temperature=temperature, deterministic=deterministic,
        )
        caption = self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)
        return caption


def plot_metrics(trainer):
    plt.plot(trainer.metrics["train_loss"], color="red", label="train loss")
    plt.plot(trainer.metrics["val_loss"], color="orange", label="valid loss")
    plt.title("loss, lower=better")
    plt.legend()
    plt.show()

    plt.plot(trainer.metrics["train_perplexity"], color="blue", label="train perplexity")
    plt.plot(trainer.metrics["val_perplexity"], color="lightblue", label="valid perplexity")
    plt.title("perplexity, lower=better")
    plt.legend()
    plt.show()


def demo_generations(trainer, val_df, n=10):
    for i in range(n):
        det = i > n - 5
        test = val_df.sample(n=1).values[0]
        test_img, test_caption = test[0], test[1]
        plt.imshow(Image.open(test_img).convert("RGB"))
        t = np.random.uniform(0.5, 1.5)
        gen_caption = trainer.generate_caption(test_img, temperature=t, deterministic=det)
        plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {t:.2f} deterministic: {det}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    train_dl, val_dl, val_df = get_dataloaders(batch_size=train_config.batch_size)

    trainer = Trainer(model_config, train_config, (train_dl, val_dl))
    trainer.fit()

    print(trainer.metrics)
    plot_metrics(trainer)

    trainer.load_best_model()
    demo_generations(trainer, val_df)
