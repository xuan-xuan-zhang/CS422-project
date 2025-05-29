import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, RocCurveDisplay)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


class IrisAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Data Analysis System")
        self.root.geometry("1000x700")

        self.default_params = {
            'test_size': 0.3,
            'random_state': 42,
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'n_clusters': 3
        }

        self.create_main_interface()
        self.load_data()
    #Create the main interface
    def create_main_interface(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        params_frame = ttk.LabelFrame(top_frame, text="Parameter settings", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(params_frame, text="Test set ratio:").grid(row=0, column=0, sticky=tk.W)
        self.test_size_entry = ttk.Entry(params_frame)
        self.test_size_entry.insert(0, str(self.default_params['test_size']))
        self.test_size_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Random seed:").grid(row=1, column=0, sticky=tk.W)
        self.random_state_entry = ttk.Entry(params_frame)
        self.random_state_entry.insert(0, str(self.default_params['random_state']))
        self.random_state_entry.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Number of decision trees :").grid(row=2, column=0, sticky=tk.W)
        self.n_estimators_entry = ttk.Entry(params_frame)
        self.n_estimators_entry.insert(0, ",".join(map(str, self.default_params['n_estimators'])))
        self.n_estimators_entry.grid(row=2, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Maximum depth :").grid(row=3, column=0, sticky=tk.W)
        self.max_depth_entry = ttk.Entry(params_frame)
        self.max_depth_entry.insert(0, ",".join(map(str, self.default_params['max_depth'])))
        self.max_depth_entry.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Minimum sample size for segmentation:").grid(row=4, column=0, sticky=tk.W)
        self.min_samples_split_entry = ttk.Entry(params_frame)
        self.min_samples_split_entry.insert(0, ",".join(map(str, self.default_params['min_samples_split'])))
        self.min_samples_split_entry.grid(row=4, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Minimum number of samples at the leaf node:").grid(row=5, column=0, sticky=tk.W)
        self.min_samples_leaf_entry = ttk.Entry(params_frame)
        self.min_samples_leaf_entry.insert(0, ",".join(map(str, self.default_params['min_samples_leaf'])))
        self.min_samples_leaf_entry.grid(row=5, column=1, sticky=tk.W)

        ttk.Label(params_frame, text="Number of clusters:").grid(row=6, column=0, sticky=tk.W)
        self.n_clusters_entry = ttk.Entry(params_frame)
        self.n_clusters_entry.insert(0, str(self.default_params['n_clusters']))
        self.n_clusters_entry.grid(row=6, column=1, sticky=tk.W)

        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        self.run_button = ttk.Button(button_frame, text="Operation analysis", command=self.run_analysis)
        self.run_button.pack(pady=5, fill=tk.X)

        self.reset_button = ttk.Button(button_frame, text="Reset parameters", command=self.reset_params)
        self.reset_button.pack(pady=5, fill=tk.X)

        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.notebook = ttk.Notebook(self.result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_data = ttk.Frame(self.notebook)
        self.tab_eda = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)
        self.tab_cluster = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_data, text="Data Overview")
        self.notebook.add(self.tab_eda, text="Exploratory analysis")
        self.notebook.add(self.tab_model, text="Model results")
        self.notebook.add(self.tab_cluster, text="Cluster analysis")

    def reset_params(self):
        self.test_size_entry.delete(0, tk.END)
        self.test_size_entry.insert(0, str(self.default_params['test_size']))

        self.random_state_entry.delete(0, tk.END)
        self.random_state_entry.insert(0, str(self.default_params['random_state']))

        self.n_estimators_entry.delete(0, tk.END)
        self.n_estimators_entry.insert(0, ",".join(map(str, self.default_params['n_estimators'])))

        self.max_depth_entry.delete(0, tk.END)
        self.max_depth_entry.insert(0, ",".join(map(str, self.default_params['max_depth'])))

        self.min_samples_split_entry.delete(0, tk.END)
        self.min_samples_split_entry.insert(0, ",".join(map(str, self.default_params['min_samples_split'])))

        self.min_samples_leaf_entry.delete(0, tk.END)
        self.min_samples_leaf_entry.insert(0, ",".join(map(str, self.default_params['min_samples_leaf'])))

        self.n_clusters_entry.delete(0, tk.END)
        self.n_clusters_entry.insert(0, str(self.default_params['n_clusters']))

    def load_data(self):
        print("Loading the dataset......")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        columns = ['Length of the calyx', 'The width of the calyx', 'Petal length', 'Petal width', 'Category']
        self.df = pd.read_csv(url, names=columns)

        self.show_data_overview()

    def show_data_overview(self):
        for widget in self.tab_data.winfo_children():
            widget.destroy()

        text = tk.Text(self.tab_data, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(self.tab_data, orient=tk.VERTICAL, command=text.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(self.tab_data, orient=tk.HORIZONTAL, command=text.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        text.configure(xscrollcommand=scroll_x.set)

        text.insert(tk.END, "Dataset information:\n")
        with pd.option_context('display.max_colwidth', None):
            info_str = self.df.info(buf=None)
            text.insert(tk.END, str(info_str) + "\n\n")

        text.insert(tk.END, "The first five rows of data:\n")
        text.insert(tk.END, str(self.df.head()) + "\n\n")

        text.insert(tk.END, "The number of missing values in each column:\n")
        text.insert(tk.END, str(self.df.isnull().sum()) + "\n")

        text.configure(state=tk.DISABLED)

    def run_analysis(self):
        try:
            params = self.get_user_params()
            self.preprocess_data(params)
            self.exploratory_data_analysis()
            self.train_and_evaluate_model(params)
            self.cluster_analysis(params)
            messagebox.showinfo("Success", "Analysis completed！")

        except Exception as e:
            messagebox.showerror("Error", f"Errors occurred during the analysis process.:\n{str(e)}")

    def get_user_params(self):
        params = {
            'test_size': float(self.test_size_entry.get()),
            'random_state': int(self.random_state_entry.get()),
            'n_estimators': [int(x) for x in self.n_estimators_entry.get().split(",")],
            'max_depth': [int(x) if x != 'None' else None for x in self.max_depth_entry.get().split(",")],
            'min_samples_split': [int(x) for x in self.min_samples_split_entry.get().split(",")],
            'min_samples_leaf': [int(x) for x in self.min_samples_leaf_entry.get().split(",")],
            'n_clusters': int(self.n_clusters_entry.get())
        }
        return params

    def preprocess_data(self, params):
        print("\nData preprocessing is currently underway....")
        self.le = LabelEncoder()
        self.df['Category'] = self.le.fit_transform(self.df['Category'])
        self.scaler = StandardScaler()
        X = self.df.drop('Category', axis=1)
        y = self.df['Category']
        self.X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, y, test_size=params['test_size'], random_state=params['random_state'])

    def exploratory_data_analysis(self):
        print("\nExploratory data analysis visualization is being generated....")

        for widget in self.tab_eda.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.tab_eda)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Feature Relationship Matrix Diagram
        frame1 = ttk.Frame(scrollable_frame)
        frame1.pack(fill=tk.X, pady=10)
        fig1 = plt.figure(figsize=(10, 8))
        sns.pairplot(pd.DataFrame(self.X_scaled, columns=self.df.columns[:-1]),
                     diag_kind='kde', corner=True)
        plt.title('Feature Relationship Matrix Diagram (Standardized Data)')
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.X)

        # Correlation heatmap of features
        frame2 = ttk.Frame(scrollable_frame)
        frame2.pack(fill=tk.X, pady=10)
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        corr_matrix = pd.DataFrame(self.X_scaled, columns=self.df.columns[:-1]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax2)
        ax2.set_title('Correlation heatmap of features')
        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.X)

        # PCA Dimensionality Reduction Visualization
        frame3 = ttk.Frame(scrollable_frame)
        frame3.pack(fill=tk.X, pady=10)
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X_scaled)
        scatter = ax3.scatter(self.X_pca[:, 0], self.X_pca[:, 1],
                              c=self.df['Category'], cmap='viridis', alpha=0.7)
        ax3.set_xlabel('Principal Component 1 (Explanation of variance: {:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100))
        ax3.set_ylabel('Principal Component 2 (Explanation of variance: {:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100))
        ax3.set_title('PCA Dimensionality Reduction Visualization (2D)')
        canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.X)

        # Elbow Rule Diagram
        frame4 = ttk.Frame(scrollable_frame)
        frame4.pack(fill=tk.X, pady=10)
        fig4 = plt.figure(figsize=(8, 6))
        ax4 = fig4.add_subplot(111)
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.X_scaled)
            wcss.append(kmeans.inertia_)
        ax4.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax4.set_xlabel('Number of clusters')
        ax4.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
        ax4.set_title('Elbow Rule (Determining the Optimal Number of Clusters)')
        canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.X)

    def train_and_evaluate_model(self, params):
        print("\nThe model is currently being trained and evaluated....")
        for widget in self.tab_model.winfo_children():
            widget.destroy()

        param_grid = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'min_samples_leaf': params['min_samples_leaf']
        }

        model = RandomForestClassifier(random_state=params['random_state'])
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        print("\nOptimal model parameters:", grid_search.best_params_)

        y_pred = self.best_model.predict(self.X_test)
        y_prob = self.best_model.predict_proba(self.X_test)

        main_frame = ttk.Frame(self.tab_model)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Confusion Matrix
        frame1 = ttk.Frame(scrollable_frame)
        frame1.pack(fill=tk.X, pady=10)
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.le.classes_, yticklabels=self.le.classes_, ax=ax1)
        ax1.set_xlabel('Prediction category')
        ax1.set_ylabel('True category')
        ax1.set_title('Confusion Matrix')
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.X)

        # Feature importance
        frame2 = ttk.Frame(scrollable_frame)
        frame2.pack(fill=tk.X, pady=10)
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        importances = self.best_model.feature_importances_
        features = self.df.columns[:-1]
        indices = np.argsort(importances)[::-1]
        ax2.bar(range(len(importances)), importances[indices], align='center')
        ax2.set_xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
        ax2.set_xlabel('Characteristics')
        ax2.set_ylabel('Importance score')
        ax2.set_title('Ranking of feature importance')
        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.X)

        # Multi-class ROC curve
        frame3 = ttk.Frame(scrollable_frame)
        frame3.pack(fill=tk.X, pady=10)
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        for i in range(len(self.le.classes_)):
            RocCurveDisplay.from_predictions(
                self.y_test == i,
                y_prob[:, i],
                name=f"Category{self.le.classes_[i]}",
                plot_chance_level=(i == 0),
                ax=ax3
            )
        ax3.set_xlabel('True Positive Rate')
        ax3.set_ylabel('True rate')
        ax3.set_title('Multi-class ROC curve')
        ax3.legend(loc="lower right")
        canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.X)

        # Classification Report
        frame4 = ttk.Frame(scrollable_frame)
        frame4.pack(fill=tk.X, pady=10)
        fig4 = plt.figure(figsize=(8, 6))
        ax4 = fig4.add_subplot(111)
        report = classification_report(self.y_test, y_pred, target_names=self.le.classes_)
        ax4.axis('off')
        ax4.text(0, 0.5, report, fontfamily='monospace', fontsize=10)
        ax4.set_title('Classification Report')
        canvas4 = FigureCanvasTkAgg(fig4, master=frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.X)

        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_prob, multi_class='ovr')

        metrics_frame = ttk.Frame(scrollable_frame)
        metrics_frame.pack(fill=tk.X, pady=10)

        ttk.Label(metrics_frame, text=f"Model accuracy rate: {accuracy:.2f}").pack(side=tk.LEFT, padx=10)
        ttk.Label(metrics_frame, text=f"Multi-category AUC score: {auc_score:.2f}").pack(side=tk.LEFT, padx=10)

    def cluster_analysis(self, params):

        print("\nA cluster analysis is currently underway....")

        for widget in self.tab_cluster.winfo_children():
            widget.destroy()

        kmeans = KMeans(n_clusters=params['n_clusters'], init='k-means++', random_state=params['random_state'])
        clusters = kmeans.fit_predict(self.X_scaled)

        main_frame = ttk.Frame(self.tab_cluster)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Visualization of clustering results
        frame1 = ttk.Frame(scrollable_frame)
        frame1.pack(fill=tk.X, pady=10)
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        scatter = ax1.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                    s=300, c='red', marker='X', label='Cluster center')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('K-means clustering result (after PCA dimensionality reduction)')
        ax1.legend()
        canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.X)

        # Clustering and comparison with the true categories
        frame2 = ttk.Frame(scrollable_frame)
        frame2.pack(fill=tk.X, pady=10)
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        scatter = ax2.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.df['Category'], cmap='viridis', alpha=0.7)
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_title('Actual category distribution')
        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.X)

        info_frame = ttk.Frame(scrollable_frame)
        info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(info_frame, text=f"Number of clusters: {params['n_clusters']}").pack(side=tk.LEFT, padx=10)
        ttk.Label(info_frame, text=f"Contour coefficient: {self.calculate_silhouette_score(clusters):.2f}").pack(side=tk.LEFT, padx=10)

    def calculate_silhouette_score(self, clusters):
        from sklearn.metrics import silhouette_score
        try:
            return silhouette_score(self.X_scaled, clusters)
        except:
            return -1


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisAnalysisApp(root)
    root.mainloop()