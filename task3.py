import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/Users/DELL/Desktop/titanic.csv')

# Set plot style
sns.set(style='whitegrid')
plt.figure(figsize=(20, 12))

# 1. PassengerId Distribution
plt.subplot(3, 4, 1)
sns.histplot(df['PassengerId'], bins=10)
plt.title('PassengerId')

# 2. Survived Distribution
plt.subplot(3, 4, 2)
sns.countplot(x='Survived', data=df)
plt.title('Survived')

# 3. Pclass Distribution
plt.subplot(3, 4, 3)
sns.countplot(x='Pclass', data=df)
plt.title('Pclass')

# 4. Name (Unique Count)
plt.subplot(3, 4, 4)
unique_names = df['Name'].nunique()
plt.text(0.5, 0.5, f'{unique_names}\nunique values', ha='center', va='center', fontsize=14)
plt.axis('off')
plt.title('Name')

# 5. Sex Distribution
plt.subplot(3, 4, 5)
sex_counts = df['Sex'].value_counts(normalize=True) * 100
sns.barplot(x=sex_counts.index, y=sex_counts.values)
plt.title('Sex')

# 6. Age Distribution
plt.subplot(3, 4, 6)
sns.histplot(df['Age'].dropna(), bins=10)
plt.title('Age')

# 7. SibSp Distribution
plt.subplot(3, 4, 7)
sns.countplot(x='SibSp', data=df)
plt.title('SibSp')

# 8. Parch Distribution
plt.subplot(3, 4, 8)
sns.countplot(x='Parch', data=df)
plt.title('Parch')

# 9. Ticket (Top 2 Categories + Other)
plt.subplot(3, 4, 9)
top_tickets = df['Ticket'].value_counts().nlargest(2)
other_count = df['Ticket'].nunique() - 2
ticket_labels = list(top_tickets.index) + ['Other']
ticket_sizes = list((top_tickets / df['Ticket'].shape[0] * 100).round(1)) + [100 - top_tickets.sum() / df['Ticket'].shape[0] * 100]
sns.barplot(x=ticket_labels, y=ticket_sizes)
plt.ylabel('Percentage')
plt.title('Ticket')

# 10. Fare Distribution
plt.subplot(3, 4, 10)
sns.histplot(df['Fare'], bins=20)
plt.title('Fare')

plt.tight_layout()
plt.show()
