# Directory Structure 📂
📂`client`: Contains the Postpartum Hemorrhage Risk Assessment Tool (.exe file), trained pkl model files, and the corresponding model parameter JSON file.

📂`dataset`: Contains simulated datasets synthesized from the original dataset (with a data distribution similar to the original dataset). These simulated datasets are implemented using the synthpop package (version v1.9.2) in R. The simulated datasets have an Accuracy of 0.8138 and an AUC of 0.8631, both of which are very close to those of the original dataset.

📂`figure`: Contains a diagram of the Postpartum Hemorrhage Risk Assessment Tool.

📂`predict`: Contains implementation code for making predictions using the trained model.

📂`train`: Contains code for model training using the dataset.


# Postpartum Hemorrhage Risk Assessment Tool

![Client](https://github.com/Zhangcantprogram/PPH_Prediction/blob/main/figure/client_figure.png?raw=true)  
A user-friendly desktop application for assessing postpartum hemorrhage (PPH) risk using machine learning.

## Key Features ✨
- **Risk Prediction**: Instant PPH risk level (Low/Medium/High) with probability percentage
- **Smart Inputs**: 
  - Predefined medical categories (Type of placental location, Mode of delivery, etc.)
  - Numerical value validation for clinical metrics
- **Model Management**:
  - Load pre-trained prediction models (.pkl)
  - Train new models with custom datasets
  - Save optimized models
- **Advanced Insights**:
  - Feature importance visualization
  - Training progress tracking
  - Performance metrics (AUC, Accuracy, Precision, Recall, F1-Score)

## System Requirements 💻
- Windows 10/11 (64-bit)

## Quick Start Guide 🚀

### For Risk Assessment:
1. **Load Model**  
   Click `Browse` to select a `.pkl` model file
   
2. **Enter Patient Data**  
   - Select categorical options from dropdowns  
   - Input numerical values in highlighted fields  
     *Example: C-reactive protein(38-40w) levels, D-dimer(29-40w) results*

3. **Get Prediction**  
   Click `Predict` to see:  
   🟢 Low Risk (≤35%)  
   🟠 Medium Risk (36%-70%)  
   🔴 High Risk (>70%)

### For Model Training:
1. **Prepare Data**  
   CSV/Excel format with:
   - 15 clinical features
   - `label` column (0=Without PPH, 1=PPH)（The example dataset is in `/dataset/dataset.csv`）
2. **Train Model**  
   - Load dataset (`Load Dataset`)
   - Start training (`Start Training`)
   - Stop training (`Stop training`)
   - Monitor progress via fold counter
3. **Save & Use**  
   - Save best model (`Save Model`)
   - View feature importance charts (`Features Importance`)
   - View model training logs (`training logs`)

4. **`Clear` Button Function**  
   Resets the training environment by removing loaded datasets, resetting parameters, clearing metrics, and deleting temporary model files. Does not affect loaded prediction models or current input values. Use when starting new training sessions or removing outdated data.

## Complete Input Guidelines 📋

### Categorical Variables
| Parameter                  | Options                                                      | Encoded Value |
| -------------------------- | ------------------------------------------------------------ | ------------- |
| Type of placental location | • Low-lying placenta<br>• Unspecified placenta previa<br>• Marginal placenta previa<br>• Partial placenta previa<br>• Complete placenta previa<br>• Pernicious placenta previa | 0-5           |
| Mode of delivery           | • Cesarean delivery<br>• Vaginal delivery                    | 0-1           |
| Perineal laceration        | • Without PL<br>• 1st-degree perineal laceration<br>• 2nd-degree perineal laceration | 0-2           |
| Placenta accreta spectrum  | • Without PAS<br>• Placenta accreta<br>• Placenta increta    | 0-2           |

### Continuous Variables (Numerical Inputs)
| Parameter                                                    | Unit   | Example Value |
| ------------------------------------------------------------ | ------ | ------------- |
| C-reactive protein(38-40w)(mg/L)                             | mg/L   | 53            |
| D-dimer(29-40w)(mg/L)                                        | mg/L   | 1.03          |
| Potassium(34+1-40w)(mmol/L)                                  | mmol/L | 3.7           |
| Total protein(38-40w)(g/L)                                   | g/L    | 53.1          |
| Serum myoglobin(32+1-40w)(ng/mL)                             | ng/mL  | 29.18         |
| Alkaline phosphatase(38-40w)(U/L)                            | U/L    | 115           |
| Blood white blood cells(38-40w)(10^9/L)                      | 10^9/L | 11.62         |
| Absolute value of lymphocytes(38-40w)(10^9/L)                | 10^9/L | 1.46          |
| Second-Trimester T18 Risk                                    | Ratio  | 1874.71       |
| Distance from placental lower edge to internal ostium(25+1-32w)(mm) | mm     | 39.93         |
| Distance from placental lower edge to internal ostium(38+1-40w)(mm) | mm     | 14            |

## Important Notes ⚠️

- Preferably, the column names and order of the dataset are the same as the examples we provided
- Training requires balanced dataset (recommended PPH cases ≥30%)
- Log files auto-save in `/training_logs`
- The recommended screen zoom ratio is **125%**
