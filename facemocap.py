import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

class FaceMoCap:
    def __init__(self, root_dir=None):
        # Initialize the FaceMoCap class
        if root_dir is None:
            self.root_dir = '/media/rodriguez/30F6-F853/Data_Mocap/organized_files_M_repeat.csv'
        else:
            self.root_dir = root_dir
        self.original_mocap_data = self.load_data_csv()
        self.valid_mocap_data = self.load_valid_mocap_data()
        self.purified_mocap_data = self.load_purified_mocap_data()
        self.processing_mocap_data = self.load_processing_mocap_data()
        self.volunteer_mocap_data = self.load_volunteer_mocap_data()
        self.patient_mocap_data = self.load_patient_mocap_data()
        
        self.markers_names = ['M1', 'M2', 'M3', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'S1', 'S2', 'S3', 'S4', 'S5', 'G0', 
        'S6', 'S7', 'S8', 'S9', 'S10', 'D1', 'D2', 'D3', 'D4', 'Ca01', 'Vl1', 'Vl2', 'Vl3', 'Ca02', 'Ca03', 
        'Vl4', 'Vl5', 'Vl6', 'Ca04', 'N3', 'N4', 'Pln7', 'Pln8', 'Pln9', 'Pln10', 'Pln11', 'R5', 'C02', 
        'Cms7', 'Cms5', 'Cms3', 'Cms1', 'C01', 'Z5', 'Z4', 'Z3', 'Z2', 'Z1', 'BZ1', 'BZ2', 'BZ3', 'BZ4', 
        'BZ5', 'Dao1', 'Dao2', 'Dao3', 'B6', 'B5', 'B4', 'B3', 'B2', 'B1', 'Cmi1', 'Cmi3', 'Cmi5', 'H3', 
        'H1', 'H0', 'M0', 'H6', 'H4', 'Dao4', 'Dao5', 'Dao6', 'B12', 'B11', 'B10', 'B9', 'B8', 'B7', 'BZ6', 
        'BZ7', 'BZ8', 'BZ9', 'BZ10', 'Z11', 'Z10', 'Z9', 'Z8', 'Z7', 'R2', 'Pln1', 'Pln2', 'Pln3', 'Pln4', 
        'Pln5', 'N1', 'N2', 'Pps01', 'Lc2', 'Pps02', 'Lc5']


    def load_processing_mocap_data(self):
        df = pd.read_csv('processing_data.csv')
        return df

    def load_purified_mocap_data(self):
        # Load the purified CSV file into a DataFrame
        df = pd.read_csv('purified_mocap_data.csv')
        return df

    def load_volunteer_mocap_data(self):
        df = pd.read_csv('volunteer_data.csv')
        return df

    def load_patient_mocap_data(self):
        df = pd.read_csv('patient_data.csv')
        return df
    
    def get_purified_mocap_data(self):
        return self.purified_mocap_data

    def get_processing_mocap_data(self):
        return self.processing_mocap_data

    def load_data_csv(self):
        # Load the updated CSV file into a DataFrame
        df = pd.read_csv(self.root_dir)
        return df
    
    # display mocap data
    def display_data(self, data='original'):
        if data == 'original':
            print(self.original_mocap_data)
        elif data == 'valid':
            print(self.valid_mocap_data)
        else:
            print("Invalid data type. Please specify 'original' or 'valid'.")
        
    def load_valid_mocap_data(self):
        # Filter the rows according to the specified conditions
        valid_data = self.original_mocap_data[
            self.original_mocap_data['facial_movement_id'].isin([f'M{i}' for i in range(1, 10)]) &  # facial_movement_id in M1, M2, ..., M9
            (self.original_mocap_data['info_state'] == 'processing') &  # info_state is 'processing'
            (self.original_mocap_data['repetitive_movement'] == False)  # repetitive_movement is False
        ]
        return valid_data
    
    def get_valid_mocap_data(self):
        return self.valid_mocap_data
    
    def get_volunteer_mocap_data(self):
        return self.volunteer_mocap_data
    
    def get_patient_mocap_data(self):
        return self.patient_mocap_data

    def get_row_from_mocap_data(self, index=0, data='valid'):
        if data == 'valid':
            df = self.valid_mocap_data
        elif data == 'original':
            df = self.original_mocap_data
        elif data == 'purified':
            df = self.purified_mocap_data
        else:
            print('Invalid data type. Please specify "valid", "purified" or "original".')

        if index ==-1:
            # Randomly select a row in df
            row = df.sample()
            return row
        elif index >= 0:
            # Select a row by index
            row = df.iloc[index]
            return row

    def csv_to_spc(self, filepath):
        # Read the CSV, skipping the first 4 rows and selecting columns 2 to 325 (0-indexed)
        df = pd.read_csv(filepath, skiprows=4, usecols=range(2, 326))
        
        # Convert the dataframe to a numpy array. The expected shape is (num_timestamps, 324)
        data = df.to_numpy()
        
        # Reshape the data into (num_timestamps, 108 markers, 3 dimensions)
        spc = data.reshape(data.shape[0], len(self.markers_names), 3)
        
        return spc
    
    def remove_support_and_eyes(self, spc):
        return spc[:, 3:-4, :]
    
    def count_complete_point_clouds(self, spc: np.ndarray) -> int:
        """
        Counts the number of complete point clouds (without any NaN values) in a sequence of point clouds.

        Parameters:
            spc (np.ndarray): A 3D numpy array of shape (N, M, 3) representing a sequence of point clouds.
        
        Returns:
            int: The number of point clouds that do not contain any NaN values.
        """
        # Create a boolean mask for each point cloud: True if there are no NaNs in that point cloud.
        valid_mask = ~np.isnan(spc).any(axis=(1, 2))
        
        # Sum the True values to get the count of complete point clouds.
        return int(np.sum(valid_mask))
  
    def determine_empty_spc(self, spc):
        if spc.size == 0:
            return True
        # Check if the entire spc array is NaN or all zeros
        return np.isnan(spc).all()

    def determine_small_spc(self, spc, threshold=100):
        # Check if the spc array is too small
        return spc.shape[0] < threshold
    
    def filter_insufficient_spc(self, save_csv=False, filename="purified_mocap_data.csv", size_threshold=50):
        """
        Iterates through valid_mocap_data and updates info_state to 'insufficient'
        if the corresponding spc is empty or too small. If there is an issue opening
        the file, info_state is set to 'corrupted'.

        Parameters:
        - save_csv (bool): Whether to save the filtered DataFrame as a CSV file.
        - filename (str): The name of the output CSV file.
        - size_threshold (int): The minimum required size for spc.
        """

        for index, row in self.valid_mocap_data.iterrows():
            try:
                print(f"Processing {row['filepath']}...")
                spc = self.csv_to_spc(row['filepath'])  # Convert CSV to spc array

                if self.determine_empty_spc(spc) or self.determine_small_spc(spc, size_threshold):
                    self.valid_mocap_data.at[index, 'info_state'] = 'insufficient'

            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")
                self.valid_mocap_data.at[index, 'info_state'] = 'corrupted'

        if save_csv:
            self.valid_mocap_data.to_csv(filename, index=False)
            print(f"Filtered data saved to {filename}")

    def count_permanently_missing_markers(self):
        # Initialize a list to hold counts for each marker
        permanently_missing_counts = [0] * len(self.markers_names)

        # Iterate through each row in the purified mocap data
        for index, row in self.purified_mocap_data.iterrows():
            try:
                # Only consider rows where info_state is 'processing'
                if row['info_state'] == 'processing':
                    # Convert the current sequence of point clouds to spc (3D points)
                    spc = self.csv_to_spc(row['filepath'])
                    
                    # Iterate through each marker
                    for i, marker_name in enumerate(self.markers_names):
                        # Check if the entire sequence for this marker is NaN (permanently missing)
                        if np.isnan(spc[:, i, :]).all():  # If all frames are NaN for this marker
                            permanently_missing_counts[i] += 1
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")

        return permanently_missing_counts

    def remove_files_from_purified_data(self, files_to_remove, save_csv=True, filename="purified_mocap_data.csv"):
        """
        Removes rows from the purified_mocap_data DataFrame that correspond to the given file paths.

        Parameters:
        - files_to_remove (list): List of file paths to be removed.
        - save_csv (bool): Whether to save the updated DataFrame to a CSV file.
        - filename (str): The name of the output CSV file.
        """
        # Remove the rows where the 'filepath' matches any of the files to remove
        self.purified_mocap_data = self.purified_mocap_data[~self.purified_mocap_data['filepath'].isin(files_to_remove)]

        # Save the updated DataFrame to a CSV file
        if save_csv:
            self.purified_mocap_data.to_csv(filename, index=False)
            print(f"Updated purified_mocap_data saved to {filename}")

    def visualize_missing_markers(self):
        """
        Visualizes the list of permanently missing markers (i.e., markers that are NaN for the entire sequence).
        """
        missing_markers = []  # List to store missing marker names

        # Iterate through each row in the purified mocap data
        for index, row in self.purified_mocap_data.iterrows():
            try:
                # Convert the current sequence of point clouds to spc (3D points)
                spc = self.csv_to_spc(row['filepath'])

                # Check each marker if it's permanently missing (NaN for all frames)
                for i, marker_name in enumerate(self.markers_names):
                    if np.isnan(spc[:, i, :]).all():  # If the entire sequence for this marker is NaN
                        missing_markers.append(marker_name)
            except Exception as e:
                print(f"Error processing {row['filepath']}: {e}")

        # Remove duplicates (if any marker is permanently missing in multiple sequences)
        missing_markers = list(set(missing_markers))

        # Visualize the missing markers
        if missing_markers:
            plt.figure(figsize=(10, 6))
            plt.bar(missing_markers, [1] * len(missing_markers))  # Use 1 for each missing marker
            plt.xlabel('Missing Markers')
            plt.ylabel('Count')
            plt.title('Permanently Missing Markers')
            plt.xticks(rotation=90)
            plt.show()
        else:
            print("No permanently missing markers found.")

    def visualize_missing_list(self, missing_list=None):
        """
        Visualizes the missing_list of permanently missing markers using a bar chart.
        
        Parameters:
        - missing_list (list, optional): A list where each element is the count of sequences in which the corresponding
                                         marker (based on self.markers_names) was permanently missing.
                                         If not provided, it calculates the missing_list by calling
                                         self.count_permanently_missing_markers().
        """
        # Calculate missing_list if not provided
        if missing_list is None:
            missing_list = self.count_permanently_missing_markers()

        # Ensure the missing_list matches the number of markers
        if len(missing_list) != len(self.markers_names):
            print("Error: The length of missing_list does not match the number of markers.")
            return

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(self.markers_names, missing_list, color='skyblue')
        plt.xlabel("Markers")
        plt.ylabel("Count of Permanently Missing Sequences")
        plt.title("Visualization of Permanently Missing Markers")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def generate_processing_data_csv(self, filename="processing_data.csv"):
        """
        Generates a CSV file containing only the rows where info_state is 'processing'.

        Parameters:
        - filename (str): The name of the output CSV file. Default is 'processing_data.csv'.
        """
        # Filter the rows where 'info_state' is 'processing'
        processing_data = self.purified_mocap_data[self.purified_mocap_data['info_state'] == 'processing']
        
        # Save the filtered data to the specified CSV file
        processing_data.to_csv(filename, index=False)
        print(f"Processing data saved to {filename}")

    def generate_volunteer_data_csv(self, filename="volunteer_data.csv"):
        """
        Generates a CSV file containing only the rows where participant_condition is 'volunteer'.

        Parameters:
        - filename (str): The name of the output CSV file. Default is 'volunteer_data.csv'.
        """
        # Filter the rows where 'participant_condition' is 'volunteer'
        volunteer_data = self.processing_mocap_data[self.processing_mocap_data['participant_condition'] == 'volunteer']
        
        # Save the filtered data to the specified CSV file
        volunteer_data.to_csv(filename, index=False)
        print(f"Volunteer data saved to {filename}")
    
    def generate_patient_data_csv(self, filename="patient_data.csv"):
        """
        Generates a CSV file containing only the rows where participant_condition is 'patient'.

        Parameters:
        - filename (str): The name of the output CSV file. Default is 'patient_data.csv'.
        """
        # Filter the rows where 'participant_condition' is 'patient'
        patient_data = self.processing_mocap_data[self.processing_mocap_data['participant_condition'] == 'patient']
        
        # Save the filtered data to the specified CSV file
        patient_data.to_csv(filename, index=False)
        print(f"Patient data saved to {filename}")
        
if __name__ == "__main__":
    # Test the FaceMoCap class
    mocap = FaceMoCap()
    volunteer_df = mocap.get_volunteer_mocap_data()
    patient_df = mocap.get_patient_mocap_data()
    print("Volunteer data:")
    print(volunteer_df)
    print("\nPatient data:")
    print(patient_df)