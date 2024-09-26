import pandas as pd

def main():
    # Read the CSV file
    df = pd.read_csv("Student_Performance.csv")

    # Replace 'Yes' and 'No' with 1.0 and 0.0 in the 'Extracurricular Activities' column
    df['Extracurricular Activities'] = df['Extracurricular Activities'].replace({'Yes': 1.0, 'No': 0.0})

    # Print the DataFrame (optional, for verification)
    print(df)

    # Convert all columns to float
    for column in df.columns:
        df[column] = df[column].astype(float)

    # Write the DataFrame back to the CSV file
    df.to_csv("Student_Performance.csv", index=False)

# Run the main function
main()