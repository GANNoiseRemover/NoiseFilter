Here you can check out the USE CASE and its description for out project.

1. Take a look at use case diagram ->
   
![Error](https://github.com/GANNoiseRemover/NoiseFilter/blob/main/img/USE_CASE1.png)


USE CASE description ->

---

## **UC0 – Power On/Off Device (Potentiometer Switch)**

* **Actor**: User
* **Preconditions**: Device is powered by battery.
* **Postconditions**: Device is either active (ON) or inactive (OFF).
* **Main Flow**:
  1. User rotates the potentiometer to the ON position.
  2. System initializes microphone, amplifier, and processing unit.
  3. User rotates potentiometer back to switch OFF when not in use.
* **Alternative Flow**:
  * If the battery is depleted, the device fails to power ON.

---

## **UC01 – Adjust Volume (Potentiometer Control)**
* **Actor**: User
* **Preconditions**: Device must be ON (**extends UC0**).
* **Postconditions**: Amplification level is changed.
* **Main Flow**:
  1. User adjusts potentiometer to increase or decrease volume.
  2. System updates amplifier gain accordingly.
* **Alternative Flow**:
  * If the device is OFF, volume adjustment has no effect.

---

## **UC1 – Capture Audio Input (Microphone)** -> Maybe its worth to make another use case diagram for detailed description of this USE CASE????
* **Actor**: System (automatic action once ON)
* **Preconditions**: Device is ON.
* **Postconditions**: Audio signal (speech + noise) is captured.
* **Main Flow**:
  1. Microphone receives surrounding sounds.
  2. Converts sound waves into electrical signals.

---

## **UC2 – Amplify Audio Signal (Hearing Aid Circuit)**
* **Actor**: System
* **Preconditions**: Audio input is available from UC1.
* **Postconditions**: Audio signal strength is increased.
* **Main Flow**:
  1. Amplifier boosts captured audio signal.
  2. Output is forwarded for noise reduction.

---

## **UC3 – Noise Reduction (GAN/Autoencoder Model)**
* **Actor**: System
* **Preconditions**: Amplified signal is available.
* **Postconditions**: Noise-reduced audio signal is produced.
* **Main Flow**:
  1. Amplified audio is processed through GAN/Autoencoder.
  2. Background noise is suppressed.
  3. Speech component is preserved.
* **Included Use Cases**:
  * **UC5 (Model Training & Improvement)** – updated model enhances noise reduction.
  * **UC6 (Evaluate Audio Quality)** – feedback improves processing quality.

---

## **UC4 – Deliver Clean Audio (Headphones/Speaker)**
* **Actor**: System
* **Preconditions**: Noise-reduced signal is available.
* **Postconditions**: User hears amplified, clear audio.
* **Main Flow**:
  1. System outputs processed audio to earphone.
  2. User perceives improved speech clarity.

---

## **UC5 – Model Training & Improvement (GAN + Deep Learning)**
* **Actor**: Development Team
* **Preconditions**: Training dataset (speech + noise samples) is available.
* **Postconditions**: Improved noise-reduction model is produced.
* **Main Flow**:
  1. Team prepares and preprocesses audio datasets.
  2. Train GAN/Autoencoder model with noise-reduction objectives.
  3. Store improved model.
  4. Deploy updated model into the device.
* **Relation**: **Included by UC3** (Noise Reduction uses the improved model).

---

## **UC6 – Evaluate Audio Quality (PESQ, STOI Metrics)**
* **Actor**: Development Team
* **Preconditions**: Model has been trained and test audio samples are available.
* **Postconditions**: Evaluation results are produced (PESQ, STOI scores).
* **Main Flow**:
  1. Team runs evaluation tests on noisy/cleaned audio pairs.
  2. Calculate PESQ and STOI metrics.
  3. Record and analyze scores.
  4. Provide feedback for further training.
* **Relation**: **Included by UC3** (Noise Reduction benefits from evaluation feedback).



