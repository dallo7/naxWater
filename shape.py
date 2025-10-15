# # import pandas as pd
# # import numpy as np
# #
# #
# # def describe_and_shape_data(file_path):
# #     """
# #     Loads a CSV file into a pandas DataFrame, describes its contents,
# #     and returns the shaped data for further analysis.
# #
# #     Args:
# #         file_path (str): The path to the CSV file.
# #
# #     Returns:
# #         pd.DataFrame: The shaped and cleaned pandas DataFrame.
# #     """
# #     try:
# #         # Load the dataset
# #         df = pd.read_csv(file_path)
# #         print("--- [1] Initial Data Loaded Successfully ---")
# #         print(f"Shape of the dataset: {df.shape}\n")
# #
# #         # --- [2] Initial Data Description ---
# #         print("--- [2] Basic Information (df.info()) ---")
# #         df.info()
# #         print("\n")
# #
# #         # --- [3] Descriptive Statistics ---
# #         print("--- [3] Descriptive Statistics (df.describe()) ---")
# #         print(df.describe(include='all'))
# #         print("\n")
# #
# #         # --- [4] Data Shaping and Cleaning ---
# #         print("--- [4] Shaping and Cleaning Data ---")
# #
# #         # Standardize column names
# #         initial_columns = df.columns.tolist()
# #         df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
# #         print(f"Standardized column names from {initial_columns} to {df.columns.tolist()}")
# #
# #         # Check for missing values
# #         missing_values = df.isnull().sum()
# #         if missing_values.sum() > 0:
# #             print("\nMissing values per column:")
# #             print(missing_values[missing_values > 0])
# #             # For demonstration, we'll fill or drop missing values.
# #             # A real-world application would require a more specific strategy.
# #             # Example: Dropping rows with any missing data.
# #             # df.dropna(inplace=True)
# #             # Example: Filling missing numerical data with the mean.
# #             # df.fillna(df.mean(numeric_only=True), inplace=True)
# #         else:
# #             print("\nNo missing values found.")
# #
# #         # Identify and handle duplicates
# #         duplicates = df.duplicated().sum()
# #         if duplicates > 0:
# #             print(f"\nFound {duplicates} duplicate rows. Removing them...")
# #             df.drop_duplicates(inplace=True)
# #             print(f"New shape after removing duplicates: {df.shape}")
# #         else:
# #             print("\nNo duplicate rows found.")
# #
# #         # Convert 'date' column to datetime objects if it exists
# #         if 'date' in df.columns:
# #             try:
# #                 df['date'] = pd.to_datetime(df['date'])
# #                 print("\n'date' column successfully converted to datetime objects.")
# #             except Exception as e:
# #                 print(f"\nError converting 'date' column: {e}")
# #
# #         print("\n--- [5] Final Shaped Data Overview ---")
# #         print(f"Final shape of the dataset: {df.shape}")
# #         print("\nFirst 5 rows of the shaped data:")
# #         print(df.head())
# #
# #         return df
# #
# #     except FileNotFoundError:
# #         print(f"Error: The file '{file_path}' was not found.")
# #         return None
# #     except Exception as e:
# #         print(f"An unexpected error occurred: {e}")
# #         return None
# #
# #
# # # Example usage:
# # if __name__ == "__main__":
# #     file_name = "dataset_Gans_run_1.CSV"
# #     shaped_df = describe_and_shape_data(file_name)
# #
# #     if shaped_df is not None:
# #         # You can now use the 'shaped_df' for your analysis, e.g., plotting.
# #         print("\nData description and shaping complete. You can now proceed with your analysis.")
#
#
# import pandas as pd
# import io
#
#
# def load_generated_data():
#     """
#     Loads the generated 100-observation dataset from a string.
#
#     Returns:
#         pd.DataFrame: A pandas DataFrame containing the generated data.
#     """
#     data_string = """date,zone,supply_area,day_of_week,time_of_supply,hours_of_supply,population_served,demand_forecast_m3,actual_consumption_m3,price_per_litre,pipe_leakage_m3,complaints_received,rainfall_mm
# 2025-01-01,Southern,Area 1,Wednesday,Morning,12,10038,485,496.91819194376245,0.1,398,18,2.5529489034173487
# 2025-01-02,Western,Area 5,Thursday,Afternoon,12,10757,529,544.6867975151416,0.10943424350882518,395,22,0.5257796754088997
# 2025-01-03,Western,Area 6,Friday,Afternoon,12,10000,654,658.3028102704657,0.10874998211807868,399,29,0.16492390130549417
# 2025-01-04,Western,Area 6,Saturday,Afternoon,12,10186,555,548.8354746351052,0.1118128527581781,399,24,2.5209733221980864
# 2025-01-05,Southern,Area 4,Sunday,Morning,12,10000,489,510.6033379657069,0.11581005898844837,398,18,1.4449830501625974
# 2025-01-06,Central,Area 5,Monday,Morning,12,10188,443,446.5056715697669,0.09880193893345892,400,21,1.9688326759755712
# 2025-01-07,Western,Area 5,Tuesday,Morning,12,10486,513,485.6706077309998,0.10864210086870028,397,25,0.3957297079237077
# 2025-01-08,Western,Area 2,Wednesday,Afternoon,12,10744,528,520.2589218698377,0.10098485741695782,392,23,0.3644026850444589
# 2025-01-09,Western,Area 2,Thursday,Morning,12,11218,639,634.3409153570395,0.10091217088264563,391,20,3.585794508492067
# 2025-01-10,Western,Area 5,Friday,Afternoon,12,10729,578,593.4735999554316,0.10041185418196614,383,21,4.409395116790586
# 2025-01-11,Southern,Area 2,Saturday,Morning,12,11082,536,544.1508933256082,0.10178345155469446,381,18,0.7302482103505668
# 2025-01-12,Western,Area 5,Sunday,Afternoon,12,10996,558,561.4284907293582,0.10080644482701859,380,21,1.554443907579893
# 2025-01-13,Central,Area 5,Monday,Afternoon,12,10657,511,507.0378051287957,0.10543261273932785,372,21,2.834469279744439
# 2025-01-14,Central,Area 6,Tuesday,Afternoon,12,11136,561,540.3533810444383,0.10526027059733857,364,22,5.204555891334659
# 2025-01-15,Central,Area 3,Wednesday,Morning,12,11145,565,584.9575971485603,0.10985226760548173,361,22,2.0628292851571216
# 2025-01-16,Western,Area 1,Thursday,Morning,12,10828,623,617.518600863071,0.11181657868770208,360,21,1.0118671691236898
# 2025-01-17,Central,Area 6,Friday,Afternoon,12,11210,580,551.4552427320092,0.10525547464010332,357,18,2.7712399226343516
# 2025-01-18,Central,Area 6,Saturday,Afternoon,12,10654,583,546.9189617260589,0.10300940316498579,355,20,3.874139988582236
# 2025-01-19,Western,Area 4,Sunday,Morning,12,11082,593,556.7196695270146,0.10842065839082725,350,19,4.484210609695029
# 2025-01-20,Western,Area 5,Monday,Afternoon,12,11422,525,540.4287860167812,0.10476495540306129,350,22,1.3854743781297594
# 2025-01-21,Western,Area 4,Tuesday,Morning,12,11029,547,562.837380182607,0.10931109069177112,347,21,1.096307994406227
# 2025-01-22,Central,Area 1,Wednesday,Afternoon,12,11048,642,654.4379361093121,0.11116279169607567,337,20,1.9669584766863695
# 2025-01-23,Central,Area 5,Thursday,Morning,12,10938,655,683.4357703819875,0.1066746813296684,332,20,1.4011400277838563
# 2025-01-24,Central,Area 5,Friday,Afternoon,12,10878,610,610.9856514732159,0.10530932264639458,331,19,1.7161678077598687
# 2025-01-25,Central,Area 2,Saturday,Afternoon,12,11452,551,556.0964149950792,0.10695123977534433,323,20,5.65588371306352
# 2025-01-26,Western,Area 3,Sunday,Morning,12,11181,643,622.0125586676885,0.10658428255963955,321,21,1.0664448208882794
# 2025-01-27,Southern,Area 6,Monday,Afternoon,12,11059,573,564.3800644349377,0.11132646636752044,314,21,3.220140702581691
# 2025-01-28,Southern,Area 4,Tuesday,Afternoon,12,11306,539,531.0601331804369,0.10373859677322964,309,21,0.7225139268840209
# 2025-01-29,Western,Area 5,Wednesday,Morning,12,11029,663,640.7513511117181,0.10620864227781747,301,23,0.3957297079237077
# 2025-01-30,Western,Area 5,Thursday,Morning,12,11135,638,620.2464168037042,0.10705018693892834,297,22,3.585794508492067
# 2025-01-31,Western,Area 5,Friday,Morning,12,11340,601,610.1652174959247,0.1042790933611867,294,22,2.0628292851571216
# 2025-02-01,Central,Area 5,Saturday,Afternoon,12,11394,629,655.4851235121404,0.10452367769931751,288,23,0.7302482103505668
# 2025-02-02,Southern,Area 4,Sunday,Afternoon,12,11422,642,630.0768461757827,0.10931109069177112,284,24,1.554443907579893
# 2025-02-03,Western,Area 5,Monday,Morning,12,11452,581,595.0315809706322,0.10444585149957723,279,21,2.834469279744439
# 2025-02-04,Central,Area 6,Tuesday,Morning,12,11000,550,560.854124976767,0.10967343242686851,273,20,5.204555891334659
# 2025-02-05,Western,Area 5,Wednesday,Afternoon,12,11681,643,622.8427942704172,0.11181657868770208,269,21,2.0628292851571216
# 2025-02-06,Central,Area 4,Thursday,Afternoon,12,11601,625,620.0805156689714,0.11306001004126742,263,22,1.0118671691236898
# 2025-02-07,Western,Area 6,Friday,Afternoon,12,11631,655,628.799797071648,0.11267414902120612,258,20,2.7712399226343516
# 2025-02-08,Western,Area 3,Saturday,Morning,12,11751,592,607.7280806141315,0.11322013894760824,251,21,3.874139988582236
# 2025-02-09,Southern,Area 3,Sunday,Afternoon,12,11649,617,617.518600863071,0.11307616687007555,246,18,4.484210609695029
# 2025-02-10,Western,Area 1,Monday,Morning,12,12059,620,637.2885918512128,0.11432483842183204,242,21,1.3854743781297594
# 2025-02-11,Western,Area 6,Tuesday,Afternoon,12,11762,606,585.8023190479718,0.11571169722883042,238,20,1.096307994406227
# 2025-02-12,Western,Area 1,Wednesday,Afternoon,12,11942,674,685.0863013854124,0.11370258163539525,233,21,1.9669584766863695
# 2025-02-13,Southern,Area 5,Thursday,Afternoon,12,12000,683,674.5262174301548,0.1137976899477815,228,21,1.4011400277838563
# 2025-02-14,Western,Area 1,Friday,Afternoon,12,12000,668,690.4907094073094,0.11545620864353844,223,19,1.7161678077598687
# 2025-02-15,Central,Area 3,Saturday,Morning,12,12294,611,619.673898083626,0.11364505190623635,221,20,5.65588371306352
# 2025-02-16,Western,Area 2,Sunday,Morning,12,12173,707,707.3917409240406,0.1121852579126442,216,19,1.0664448208882794
# 2025-02-17,Western,Area 2,Monday,Morning,12,12204,614,643.0853526938221,0.11574366601449411,210,19,3.220140702581691
# 2025-02-18,Central,Area 5,Tuesday,Afternoon,12,12000,654,670.612543958933,0.1131103606834114,206,19,0.7225139268840209
# 2025-02-19,Southern,Area 3,Wednesday,Morning,12,12179,715,690.627725916962,0.11598991820692558,200,21,0.3957297079237077
# 2025-02-20,Western,Area 1,Thursday,Afternoon,12,12328,675,649.3797801833596,0.11516597711425008,197,20,3.585794508492067
# 2025-02-21,Central,Area 5,Friday,Afternoon,12,12239,692,723.238914614059,0.11478524497672288,192,20,2.0628292851571216
# 2025-02-22,Central,Area 3,Saturday,Morning,12,12269,705,713.8812679236442,0.11667000213197576,187,19,0.7302482103505668
# 2025-02-23,Western,Area 3,Sunday,Morning,12,12497,640,642.4839846660195,0.11494666635836262,183,18,1.554443907579893
# 2025-02-24,Western,Area 5,Monday,Afternoon,12,12349,690,681.6508006832604,0.11737803734080133,179,21,2.834469279744439
# 2025-02-25,Central,Area 6,Tuesday,Afternoon,12,12356,664,635.4542691517726,0.11417537300306145,175,20,5.204555891334659
# 2025-02-26,Western,Area 3,Wednesday,Morning,12,12507,678,662.6631627931327,0.11584852084992925,170,22,2.0628292851571216
# 2025-02-27,Central,Area 5,Thursday,Morning,12,12435,662,642.493902342523,0.11634789547051908,166,21,1.0118671691236898
# 2025-02-28,Central,Area 6,Friday,Afternoon,12,12543,637,649.0768407481179,0.1186725287310582,163,22,2.7712399226343516
# 2025-03-01,Central,Area 4,Saturday,Afternoon,12,12453,678,666.9691370213038,0.11746237257962663,158,21,3.874139988582236
# 2025-03-02,Western,Area 5,Sunday,Morning,12,12581,617,624.9080708687796,0.11728108428676231,155,20,4.484210609695029
# 2025-03-03,Western,Area 1,Monday,Morning,12,12791,659,680.5739327883582,0.1189441285038317,150,21,1.3854743781297594
# 2025-03-04,Western,Area 6,Tuesday,Afternoon,12,12821,692,674.6540131435868,0.1194364009774648,146,21,1.096307994406227
# 2025-03-05,Southern,Area 1,Wednesday,Afternoon,12,12929,742,752.4745300970335,0.11890306233158385,142,21,1.9669584766863695
# 2025-03-06,Southern,Area 4,Thursday,Afternoon,12,12975,720,723.3364491764614,0.11874213009590623,137,20,1.4011400277838563
# 2025-03-07,Western,Area 4,Friday,Morning,12,12999,704,704.9786419779331,0.12061448684728517,133,21,1.7161678077598687
# 2025-03-08,Central,Area 3,Saturday,Morning,12,13038,716,730.016335133602,0.1197825000578619,129,20,5.65588371306352
# 2025-03-09,Southern,Area 5,Sunday,Afternoon,12,13057,654,642.4839846660195,0.11831804928182281,126,19,1.0664448208882794
# 2025-03-10,Western,Area 3,Monday,Afternoon,12,13123,674,657.493902342523,0.12166540608253107,121,20,3.220140702581691
# 2025-03-11,Central,Area 1,Tuesday,Morning,12,13220,637,658.0558197771743,0.1215448152597274,118,20,0.7225139268840209
# 2025-03-12,Western,Area 6,Wednesday,Afternoon,12,13250,715,695.5348880785023,0.12211623548970878,114,21,0.3957297079237077
# 2025-03-13,Southern,Area 1,Thursday,Afternoon,12,13295,739,737.9542618274714,0.1205391307612658,110,20,3.585794508492067
# 2025-03-14,Central,Area 2,Friday,Morning,12,13220,729,743.0853526938221,0.12209749176378953,107,21,2.0628292851571216
# 2025-03-15,Central,Area 2,Saturday,Afternoon,12,13421,757,746.2081121171804,0.12028114639912852,103,20,0.7302482103505668
# 2025-03-16,Southern,Area 2,Sunday,Morning,12,13511,691,667.4333068936306,0.12051662580796333,100,19,1.554443907579893
# 2025-03-17,Southern,Area 6,Monday,Afternoon,12,13539,634,610.1504936338426,0.12328212176461239,96,21,2.834469279744439
# 2025-03-18,Central,Area 1,Tuesday,Afternoon,12,13601,658,666.0886192131238,0.1226926955799971,93,21,5.204555891334659
# 2025-03-19,Southern,Area 6,Wednesday,Morning,12,13627,725,712.9137812835974,0.1256372570077382,90,19,2.0628292851571216
# 2025-03-20,Central,Area 3,Thursday,Morning,12,13678,753,733.9238318187895,0.12654313524855903,87,20,1.0118671691236898
# 2025-03-21,Central,Area 4,Friday,Morning,12,13702,746,739.7542618274714,0.1251910609363533,83,21,2.7712399226343516
# 2025-03-22,Western,Area 2,Saturday,Afternoon,12,13745,699,692.7093282216668,0.12327453396956637,80,18,3.874139988582236
# 2025-03-23,Southern,Area 5,Sunday,Afternoon,12,13809,726,717.1856754320761,0.1264629471714155,77,18,4.484210609695029
# 2025-03-24,Central,Area 5,Monday,Afternoon,12,13840,713,677.2885918512128,0.12450849303640277,74,18,1.3854743781297594
# 2025-03-25,Western,Area 3,Tuesday,Morning,12,13880,683,674.5262174301548,0.1287950392683057,71,19,1.096307994406227
# 2025-03-26,Central,Area 6,Wednesday,Afternoon,12,13916,750,732.0863013854124,0.12873138870349344,68,18,1.9669584766863695
# 2025-03-27,Central,Area 5,Thursday,Afternoon,12,13962,773,767.9542618274714,0.12988891583095318,66,19,1.4011400277838563
# 2025-03-28,Western,Area 1,Friday,Morning,12,14022,764,749.5222045610582,0.12863914800880018,63,18,1.7161678077598687
# 2025-03-29,Western,Area 5,Saturday,Afternoon,12,13435,504,510.9758202711431,0.1451933425157236,66,3,9.438718796104105
# 2025-03-30,Southern,Area 2,Sunday,Morning,12,15136,768,753.4995189351595,0.1379290321820521,70,6,6.7614309519403415
# 2025-03-31,Central,Area 3,Monday,Afternoon,12,15327,693,724.1997774293084,0.1429217064231936,75,3,14.34577209778859
# 2025-04-01,Western,Area 3,Tuesday,Morning,12,13946,816,788.5634875492644,0.14109600383533266,57,4,10.282962592546568
# 2025-04-02,Western,Area 4,Wednesday,Afternoon,12,14054,775,768.612543958933,0.1413813876766467,54,4,12.759902888241088
# 2025-04-03,Western,Area 4,Thursday,Morning,12,14092,804,785.679177897931,0.1437146522919914,51,3,14.739762143787711
# 2025-04-04,Central,Area 1,Friday,Afternoon,12,14134,758,740.0631627931327,0.14309289299468167,48,5,15.526978411641775
# 2025-04-05,Central,Area 5,Saturday,Afternoon,12,14170,828,819.5398606085188,0.14250266205763952,45,4,15.586617945763742
# 2025-04-06,Western,Area 6,Sunday,Morning,12,14227,761,770.8124991195655,0.14500140645050867,42,4,14.86877197021175
# 2025-04-07,Central,Area 6,Monday,Morning,12,14279,788,781.4333068936306,0.14417537300306145,40,6,12.449195045435948
# 2025-04-08,Western,Area 5,Tuesday,Afternoon,12,14349,796,783.5684693259838,0.14407616687007555,38,3,9.584284521746452
# 2025-04-09,Central,Area 6,Wednesday,Afternoon,12,14408,829,833.0039775073404,0.1450259885873919,36,5,6.790937666992644
# 2025-04-10,Western,Area 2,Thursday,Afternoon,12,14457,756,766.4562095874288,0.14406169992487437,34,5,4.722513926884021
# 2025-04-11,Southern,Area 3,Friday,Afternoon,12,14510,794,801.3533810444383,0.14539121571408848,32,4,1.8688326759755712
# 2025-04-12,Western,Area 1,Saturday,Afternoon,12,14561,840,830.5517904033328,0.1444364009774648,30,4,0.1554443907579893
# 2025-04-13,Southern,Area 5,Sunday,Afternoon,12,14605,798,785.8023190479718,0.14361448684728517,29,5,0.4856012353355529
# 2025-04-14,Central,Area 5,Monday,Afternoon,12,14674,807,819.0886192131238,0.14455610860534298,28,4,1.4057863116900115
# 2025-04-15,Central,Area 2,Tuesday,Morning,12,14734,815,785.3917409240406,0.14660370428989376,27,5,2.155294890341735
# 2025-04-16,Central,Area 5,Wednesday,Morning,12,14781,770,767.238914614059,0.14717537300306145,26,4,3.1555512069255747
# 2025-04-17,Central,Area 1,Thursday,Afternoon,12,14842,852,836.3115456114175,0.14798991820692558,25,4,2.9463200236113277
# 2025-04-18,Central,Area 5,Friday,Afternoon,12,14890,834,845.8926941018512,0.1467814981881845,24,3,3.012586241372554
# 2025-04-19,Southern,Area 4,Saturday,Afternoon,12,14948,817,801.673898083626,0.14856708799480665,24,3,2.463661109403848
# 2025-04-20,Western,Area 3,Sunday,Morning,12,14995,856,861.9080708687796,0.14998114639912852,23,5,1.9688326759755712
# 2025-04-21,Central,Area 6,Monday,Afternoon,12,15059,829,835.9174092404066,0.14952528148834925,22,4,0.1554443907579893
# 2025-04-22,Southern,Area 1,Tuesday,Afternoon,12,15104,800,810.0558197771743,0.14755169722883042,21,3,0.4856012353355529
# 2025-04-23,Western,Area 5,Wednesday,Morning,12,15174,862,873.3542691517726,0.14761448684728517,21,4,1.4057863116900115
# 2025-04-24,Western,Area 4,Thursday,Morning,12,15220,854,846.8058197771743,0.1492166579626359,20,5,2.155294890341735
# 2025-04-25,Central,Area 5,Friday,Morning,12,15286,819,837.9542618274714,0.1488210334860012,19,4,3.1555512069255747
# 2025-04-26,Western,Area 5,Saturday,Afternoon,12,15330,865,855.7093282216668,0.15000940316498579,19,3,2.9463200236113277
# 2025-04-27,Southern,Area 4,Sunday,Afternoon,12,15394,849,864.0853526938221,0.14949121571408848,18,3,3.012586241372554
# 2025-04-28,Central,Area 1,Monday,Afternoon,12,15437,838,829.3517904033328,0.15093557989066632,17,4,2.463661109403848
# 2025-04-29,Central,Area 6,Tuesday,Afternoon,12,15510,879,887.6749911956555,0.1511103606834114,16,5,1.9688326759755712
# 2025-04-30,Central,Area 3,Wednesday,Morning,12,15559,862,842.6631627931327,0.1506516132009212,16,4,0.1554443907579893
# 2025-05-01,Central,Area 4,Thursday,Morning,12,15629,889,878.6749911956555,0.15237803734080133,15,3,0.4856012353355529
# 2025-05-02,Southern,Area 5,Friday,Morning,12,15684,879,891.9080708687796,0.15243138870349344,14,3,1.4057863116900115
# 2025-05-03,Western,Area 3,Saturday,Afternoon,12,15757,899,906.9181919437625,0.1540645050865516,13,5,2.155294890341735
# 2025-05-04,Central,Area 5,Sunday,Afternoon,12,15802,844,850.7513511117181,0.15109749176378953,13,4,3.1555512069255747
# 2025-05-05,Western,Area 4,Monday,Afternoon,12,15873,908,899.7542618274714,0.15263914800880018,12,4,2.9463200236113277
# 2025-05-06,Southern,Area 6,Tuesday,Afternoon,12,15926,878,891.3115456114175,0.15349121571408848,11,4,3.012586241372554
# 2025-05-07,Central,Area 6,Wednesday,Morning,12,15998,903,892.6749911956555,0.15402528148834925,11,4,2.463661109403848
# 2025-05-08,Central,Area 1,Thursday,Morning,12,16060,865,888.6508006832604,0.15403557989066632,10,5,1.9688326759755712
# 2025-05-09,Western,Area 5,Friday,Morning,12,16110,891,894.2081121171804,0.15535804928182281,9,4,0.1554443907579893
# 2025-05-10,Western,Area 1,Saturday,Afternoon,12,16175,910,917.1856754320761,0.15566540608253107,9,4,0.4856012353355529
# 2025-05-11,Southern,Area 3,Sunday,Afternoon,12,16223,880,880.6540131435868,0.1549448152597274,8,4,1.4057863116900115
# 2025-05-12,Western,Area 2,Monday,Afternoon,12,16291,926,904.5222045610582,0.15610849303640277,8,5,2.155294890341735
# 2025-05-13,Southern,Area 5,Tuesday,Morning,12,16340,919,926.8926941018512,0.15593557989066632,7,4,3.1555512069255747
# 2025-05-14,Central,Area 5,Wednesday,Afternoon,12,16408,941,940.6749911956555,0.15652528148834925,7,4,2.9463200236113277
# 2025-05-15,Central,Area 2,Thursday,Afternoon,12,16457,896,903.3542691517726,0.1568210334860012,6,4,3.012586241372554
# 2025-05-16,Central,Area 5,Friday,Morning,12,16524,942,933.4995189351595,0.15786391480088002,6,5,2.463661109403848
# 2025-05-17,Southern,Area 4,Saturday,Morning,12,16568,913,908.5517904033328,0.15755169722883042,5,4,1.9688326759755712
# 2025-05-18,Western,Area 3,Sunday,Afternoon,12,16639,940,944.4379361093121,0.15858632617711925,5,4,0.1554443907579893
# 2025-05-19,Central,Area 6,Monday,Afternoon,12,16688,908,918.0631627931327,0.15809749176378953,5,4,0.4856012353355529
# 2025-05-20,Southern,Area 1,Tuesday,Morning,12,16758,954,949.7093282216668,0.1593557989066632,4,4,1.4057863116900115
# 2025-05-21,Western,Area 5,Wednesday,Afternoon,12,16805,925,935.4542691517726,0.1588210334860012,4,4,2.155294890341735
# 2025-05-22,Western,Area 4,Thursday,Afternoon,12,16875,960,958.0558197771743,0.1601103606834114,4,5,3.1555512069255747
# 2025-05-23,Central,Area 5,Friday,Morning,12,16922,936,944.3517904033328,0.15993557989066632,3,4,2.9463200236113277
# 2025-05-24,Western,Area 5,Saturday,Morning,12,16995,972,968.6540131435868,0.1606516132009212,3,5,3.012586241372554
# 2025-05-25,Southern,Area 4,Sunday,Afternoon,12,17042,943,950.4839846660195,0.1605391307612658,3,4,2.463661109403848
# 2025-05-26,Central,Area 1,Monday,Afternoon,12,17112,987,975.3917409240406,0.16140645050865516,3,4,1.9688326759755712
# 2025-05-27,Central,Area 6,Tuesday,Morning,12,17164,960,961.5398606085188,0.16110849303640277,2,5,0.1554443907579893
# 2025-05-28,Central,Area 3,Wednesday,Afternoon,12,17231,1001,987.9786419779331,0.16250266205763952,2,4,0.4856012353355529
# 2025-05-29,Central,Area 4,Thursday,Afternoon,12,17277,972,969.5222045610582,0.16223780373408012,2,4,1.4057863116900115
# 2025-05-30,Southern,Area 5,Friday,Afternoon,12,17346,1016,1000.3533810444383,0.1633580492818228,1,4,2.155294890341735
# 2025-05-31,Western,Area 3,Saturday,Morning,12,17392,986,988.4357703819875,0.16309749176378953,1,4,3.1555512069255747
# 2025-06-01,Central,Area 5,Sunday,Morning,12,17462,1028,1016.5517904033328,0.16436400977464808,1,5,2.9463200236113277
# 2025-06-02,Western,Area 4,Monday,Morning,12,17511,999,997.0886192131238,0.16410849303640277,1,4,3.012586241372554
# 2025-06-03,Southern,Area 6,Tuesday,Afternoon,12,17578,1039,1024.7093282216668,0.16523780373408012,1,5,2.463661109403848
# 2025-06-04,Central,Area 6,Wednesday,Afternoon,12,17622,1009,1011.0853526938221,0.16499811463991285,1,4,1.9688326759755712
# 2025-06-05,Central,Area 1,Thursday,Afternoon,12,17694,1048,1047.8926941018512,0.16610667468132967,1,4,0.1554443907579893
# 2025-06-06,Western,Area 5,Friday,Morning,12,17740,1019,1018.8926941018512,0.16583914800880018,1,5,0.4856012353355529
# 2025-06-07,Western,Area 1,Saturday,Morning,12,17812,1059,1052.016335133602,0.1669448152597274,1,4,1.4057863116900115
# 2025-06-08,Southern,Area 3,Sunday,Afternoon,12,17855,1029,1021.0558197771743,0.16666540608253107,0,4,2.155294890341735
# 2025-06-09,Western,Area 2,Monday,Afternoon,12,17926,1069,1067.8913781283597,0.16786391480088002,0,5,3.1555512069255747
# 2025-06-10,Southern,Area 5,Tuesday,Morning,12,17978,1039,1033.4995189351595,0.16755169722883042,0,4,2.9463200236113277
# 2025-06-11,Central,Area 5,Wednesday,Afternoon,12,18049,1079,1072.0631627931327,0.1687146522919914,0,4,3.012586241372554
# 2025-06-12,Central,Area 2,Thursday,Afternoon,12,18095,1048,1051.4333068936306,0.1685391307612658,0,5,2.463661109403848
# 2025-06-13,Central,Area 5,Friday,Morning,12,18165,1088,1093.0039775073404,0.16952528148834925,0,4,1.9688326759755712
# 2025-06-14,Southern,Area 4,Saturday,Morning,12,18214,1058,1059.238914614059,0.1692166579626359,0,4,0.1554443907579893
# 2025-06-15,Western,Area 3,Sunday,Afternoon,12,18285,1098,1092.8124991195655,0.1703580492818228,0,5,0.4856012353355529
# 2025-06-16,Central,Area 6,Monday,Afternoon,12,18335,1067,1066.8058197771743,0.17000940316498579,0,4,1.4057863116900115
# 2025-06-17,Southern,Area 1,Tuesday,Morning,12,18406,1107,1105.7513511117181,0.1711103606834114,0,4,2.155294890341735
# 2025-06-18,Western,Area 5,Wednesday,Afternoon,12,18459,1076,1086.3115456114175,0.17083914800880018,0,5,3.1555512069255747
# 2025-06-19,Western,Area 4,Thursday,Afternoon,12,18531,1116,1116.6540131435868,0.1719448152597274,0,5,2.9463200236113277
# 2025-06-20,Central,Area 5,Friday,Morning,12,18579,1085,1089.4357703819875,0.17166540608253107,0,4,3.012586241372554
# 2025-06-21,Western,Area 5,Saturday,Morning,12,18651,1125,1125.7513511117181,0.17281146399128526,0,5,2.463661109403848
# 2025-06-22,Southern,Area 4,Sunday,Afternoon,12,18699,1093,1098.8124991195655,0.17255169722883042,0,4,1.9688326759755712
# 2025-06-23,Central,Area 1,Monday,Afternoon,12,18770,1133,1125.6631627931327,0.1736725287310582,0,5,0.1554443907579893
# 2025-06-24,Central,Area 6,Tuesday,Afternoon,12,18820,1102,1108.9786419779331,0.1733580492818228,0,4,0.4856012353355529
# 2025-06-25,Central,Area 3,Wednesday,Morning,12,18890,1142,1142.3533810444383,0.17449811463991285,0,5,1.4057863116900115
# 2025-06-26,Central,Area 4,Thursday,Morning,12,18942,1111,1117.8913781283597,0.17417537300306145,0,4,2.155294890341735
# 2025-06-27,Southern,Area 5,Friday,Morning,12,19012,1150,1139.7542618274714,0.17523780373408012,0,4,3.1555512069255747
# 2025-06-28,Western,Area 3,Saturday,Afternoon,12,19062,1119,1122.016335133602,0.17499811463991285,0,5,2.9463200236113277
# 2025-06-29,Central,Area 5,Sunday,Afternoon,12,19133,1158,1158.0558197771743,0.17610667468132967,0,4,3.012586241372554
# 2025-06-30,Western,Area 4,Monday,Afternoon,12,19184,1127,1126.3115456114175,0.17583914800880018,0,5,2.463661109403848
# 2025-07-01,Southern,Area 6,Tuesday,Morning,12,19253,1166,1156.4995189351595,0.1769448152597274,0,4,1.9688326759755712
# 2025-07-02,Central,Area 6,Wednesday,Afternoon,12,19305,1135,1140.7513511117181,0.17666540608253107,0,4,0.1554443907579893
# 2025-07-03,Central,Area 1,Thursday,Afternoon,12,19374,1174,1163.4995189351595,0.17786391480088002,0,5,0.4856012353355529
# 2025-07-04,Western,Area 5,Friday,Morning,12,19426,1143,1140.5222045610582,0.17755169722883042,0,4,1.4057863116900115
# 2025-07-05,Western,Area 1,Saturday,Morning,12,19495,1182,1177.3917409240406,0.1787146522919914,0,4,2.155294890341735
# 2025-07-06,Southern,Area 3,Sunday,Afternoon,12,19546,1151,1158.0631627931327,0.17840645050865516,0,5,3.1555512069255747
# 2025-07-07,Western,Area 2,Monday,Afternoon,12,19614,1190,1193.0039775073404,0.17949811463991285,0,5,2.9463200236113277
# 2025-07-08,Southern,Area 5,Tuesday,Morning,12,19665,1158,1166.8913781283597,0.1792166579626359,0,4,3.012586241372554
# 2025-07-09,Central,Area 5,Wednesday,Afternoon,12,19734,1197,1192.4995189351595,0.1803580492818228,0,5,2.463661109403848
# """
#
#     # Use io.StringIO to read the string as if it were a file
#     df = pd.read_csv(io.StringIO(data_string))
#
#     df.to_csv('dataset_Gans_run_1.csv', index=False)
#
#     return df
#
#
# # Example usage:
# if __name__ == "__main__":
#     generated_df = load_generated_data()
#     print("DataFrame loaded successfully from string.")
#     print("Shape:", generated_df.shape)
#     print("\nFirst 5 rows:")
#     print(generated_df.head())
#     print("\nData types:")
#     print(generated_df.dtypes)


import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_generated_data():
    """
    Loads a dataset with observations for every day between 2022-01-01 and 2024-08-21.
    The data logging distinction is only in the timeframes.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the generated data.
    """
    # Define the new start and end dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 8, 21)

    # Generate a date range that includes all days between the start and end dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    num_observations = len(dates)

    # Define categorical options
    zones = ['Central', 'Southern', 'Western']
    supply_areas = ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5', 'Area 6']
    # The only distinction is the timeframes of the data logging.
    time_of_supply = ['Morning', 'Afternoon']
    hours_of_supply = 12

    # Generate data for columns
    df = pd.DataFrame({
        'date': dates,
        'zone': np.random.choice(zones, num_observations),
        'supply_area': np.random.choice(supply_areas, num_observations),
        'day_of_week': [d.strftime('%A') for d in dates],
        'time_of_supply': np.random.choice(time_of_supply, num_observations),
        'hours_of_supply': hours_of_supply,
    })

    # --- General Increasing Trend ---
    df['population_served'] = np.linspace(10000, 15000, num_observations) + np.random.normal(0, 500, num_observations)
    df['population_served'] = df['population_served'].astype(int).clip(lower=10000)

    df['demand_forecast_m3'] = (df['population_served'] * 0.05) + np.sin(
        np.arange(num_observations) * np.pi / 15) * 200 + np.random.normal(0, 50, num_observations)
    df['demand_forecast_m3'] = df['demand_forecast_m3'].astype(int).clip(lower=0)

    df['actual_consumption_m3'] = df['demand_forecast_m3'] * np.random.uniform(0.95, 1.05, num_observations)
    df['actual_consumption_m3'] = df['actual_consumption_m3'].astype(float).clip(lower=0)

    df['price_per_litre'] = np.linspace(0.1, 0.15, num_observations) + np.random.normal(0, 0.005, num_observations)
    df['price_per_litre'] = df['price_per_litre'].astype(float).clip(lower=0.1)

    # --- General Decreasing Trend (Drops) ---
    df['pipe_leakage_m3'] = np.exp(np.linspace(6, 4, num_observations)) + np.random.normal(0, 5, num_observations)
    df['pipe_leakage_m3'] = df['pipe_leakage_m3'].astype(int).clip(lower=0)

    def complaints_trend(n):
        if n < 30: return 20 + np.random.normal(0, 3)
        if 30 <= n < 60: return 10 + np.random.normal(0, 2)
        return 5 + np.random.normal(0, 1)

    df['complaints_received'] = [complaints_trend(i) for i in range(num_observations)]
    df['complaints_received'] = df['complaints_received'].astype(int).clip(lower=0)

    # --- Other Data ---
    df['rainfall_mm'] = np.abs(np.sin(np.arange(num_observations) * np.pi / 20) * 10) + np.random.normal(0, 2,
                                                                                                         num_observations)
    df['rainfall_mm'] = df['rainfall_mm'].astype(float).clip(lower=0)

    df_cleaned = df.copy()
    column_name = 'date'
    df_cleaned[column_name] = pd.to_datetime(df_cleaned[column_name], errors='coerce')
    df_cleaned.dropna(subset=[column_name], inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)

    df.to_csv('dataset_Gans_run_1.csv', index=False)

    return df


# Example usage:
if __name__ == "__main__":
    generated_df = load_generated_data()
    print(
        f"DataFrame loaded successfully with dates from {generated_df['date'].min()} to {generated_df['date'].max()}.")
    print("Number of observations:", len(generated_df))
    print("\nFirst 5 rows:")
    print(generated_df.head())
    print("\nData types:")
    print(generated_df.dtypes)
