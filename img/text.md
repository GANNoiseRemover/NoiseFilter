Here you can check out the USE CASE and its description for out project.

# **1. USE CASE diagrams ->**
   
![Error](https://github.com/GANNoiseRemover/NoiseFilter/blob/main/img/USE_CASE1.png)
![Error](https://github.com/GANNoiseRemover/NoiseFilter/blob/main/img/USE_CASE2.png)

---
# **2.USE CASE description ->**
---
#  **Hearing Aid Device (Runtime)**
---
## **UC0 - Power On/Off Device (Potentiometer)**

**Actor**: User

**Description**: The user powers the hearing aid device on or off using a potentiometer switch.

**Preconditions**:

  * Device is available and functional.
  * Battery is charged or device is connected to a power source.
    
**Postconditions**:

  * If powered ON → device is ready for audio capture.
  * If powered OFF → device stops all functions.
    
**Main Flow**:

  1. User rotates potentiometer to switch ON/OFF.
  2. Device verifies power state.
  3. Indicator confirms the action (LED light).

**Alternative Flows / Extensions:**

* If battery is low → device fails to power ON and notifies user (LED).
* If potentiometer is faulty → device does not respond; requires maintenance.

---

## **UC1 -  Adjust Volume (Potentiometer)**

**Actor**: User
  
 **Description**: The user adjusts the amplification level of the clean audio via the potentiometer.
  
**Preconditions**:

  * Device must be powered ON.
    
 **Postconditions**:

  * Amplified clean audio is delivered at the chosen volume level.
    
  **Main Flow**:

  1. User rotates potentiometer to adjust gain.
  2. Device modifies amplification parameter.
  3. Clean audio is delivered at adjusted volume.

 **Alternative Flows / Extensions:**
* Potentiometer malfunction → volume remains unchanged.

---

## **UC2 - Capture Audio Input (Microphone)**

 **Actor**: User (indirectly triggers by turning device ON) / System.

 **Description**: The device captures environmental audio via the microphone for processing.
 
 **Preconditions**:

  * Device must be powered ON.
  * Microphone is functional.
    
 **Postconditions**:

  * Raw audio signal is captured and ready for noise reduction and amplification.
    
  **Main Flow**:

  1. Microphone receives sound waves.
  2. Device digitizes the input into audio signal.
  3. Signal is forwarded to the AI model for noise reduction.

  **Alternative Flows / Extensions:**
     * Microphone damaged or blocked → no input captured; device alerts user.

---

## **UC3 -  Deliver Clean Audio (Amplified + Noise-Reduced)**

 **Actor**: System
 
 **Description**: The device delivers noise-reduced and amplified audio to the user through the headphones.
 
**Preconditions**:

  * Captured audio input available.
  * Pre-trained and optimized AI model deployed on device.
    
  **Postconditions**:

  * User receives enhanced audio with reduced background noise and adjusted volume.
    
  **Main Flow**:

  1. Captured audio signal is processed by deployed AI model.
  2. Noise reduction and amplification are applied.
  3. Processed clean audio is transmitted to the speaker.
  4. User hears improved audio output.

**Alternative Flows / Extensions:**
     
* Model fails to process audio -> go to "Preparation for Deliver Clean Audio" stage.

---



---
#  **Preparation for Deliver Clean Audio**

## **UC1 -Collect Data (Clean + Noisy Recordings)**

**Actor:** Development Team

**Description:** Collect audio recordings with noise (from hearing aid device) and clean audio recordings necessary for training and testing the noise reduction model.

**Preconditions:**
* Developers have access to audio sources.
* Tools for recording and storing audio are ready.
  
**Postconditions:**
* Training and test audio data are collected and stored in a usable format.
  
**Main Flow:**
1. Identify sources of audio recordings (various noise conditions, different speakers).
2. Record or collect existing audio data.
3. Classify audio as “clean” or “noisy.”
4. Store the data for model training and testing.

**Alternative Flows / Extensions:**
* If recordings contain excessive noise, discard or re-record them.
* If insufficient data is collected, repeat the data collection process.

---

## **US2 - Model Training & Improvement (GAN (Noise Reduction) + Autoencoder)**

**Actor:** Development Team

**Description:** Develop, train, and apply the AI model (GAN + Autoencoder) for noise reduction, generating processed audio for quality evaluation.

**Preconditions:**
* Training and test data are collected (UC1).
* Developers have prepared code and model algorithms.
  
**Postconditions:**
* The trained model is capable of reducing noise on audio samples.
* Processed audio is generated for evaluation.
  
**Main Flow:**
1. Implement the code for the model (GAN + Autoencoder).
2. Load collected training data.
3. Train the model using training data.
4. Apply the trained model to test audio for noise reduction.
5. Adjust model parameters based on preliminary results to optimize performance.
   
**Alternative Flows / Extensions:**
* If the model fails to train properly, adjust hyperparameters or model architecture.
* If processed audio is unsatisfactory, revisit training with additional data or parameter tuning.

---

## **US3 - Evaluate Audio Quality (PESQ, STOI)**

**Actor:** Development Team

**Description:** Assess the quality of the processed audio generated by the trained model using PESQ and STOI metrics.

**Preconditions:**
* Model is trained and applied to test audio (UC2).
* Test audio is available.
  
**Postconditions:**
* Quality metrics (PESQ, STOI) are obtained, indicating the effectiveness of the model.
* Decision is made whether the model is ready for deployment or needs improvement.
  
**Main Flow:**
1. Load the original test audio and the test audio processed by the trained model.
2. Compute PESQ and STOI metrics for both versions of audio.
3. Analyze the results to determine model performance.
   
**Alternative Flows / Extensions:**
* If metrics are below acceptable thresholds, return to UC2 to retrain or adjust the model.

---

## **US4 - Deploy Optimized Model (to Runtime Device)**

**Actor:** Development Team

**Description:** Deploy the trained, tested, and optimized model to the runtime hearing aid device for end-user operation.

**Preconditions:**
* Model is trained, applied to test audio, and evaluated (UC2 & UC3).
* Runtime device is ready for deployment.
  
**Postconditions:**
* AI model is successfully deployed on the device.
* Deliver Clean Audio functionality is ready for use by the user.
  
**Main Flow:**
1. Prepare the model for deployment (conversion, packaging).
2. Transfer the model to the runtime device.
3. Verify correct installation and functionality of the deployed model.
   
**Alternative Flows / Extensions:**
* If deployment fails, fix errors and repeat the deployment process.



