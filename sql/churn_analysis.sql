CREATE TABLE churn_data(
	CustomerID VARCHAR(20) PRIMARY KEY,
	Gender VARCHAR(10),	SeniorCitizen INT, Partner VARCHAR(15), Dependents VARCHAR(15),	
	Tenure INT,	PhoneService VARCHAR(15), MultipleLines VARCHAR(15),
	InternetService VARCHAR(15), OnlineSecurity VARCHAR(15), OnlineBackup VARCHAR(15), DeviceProtection VARCHAR(15), TechSupport VARCHAR(15),
	StreamingTV VARCHAR(15), StreamingMovies VARCHAR(15),
	Contract VARCHAR(15), PaperlessBilling VARCHAR(15), PaymentMethod VARCHAR(20), MonthlyCharges DECIMAL(10,2), TotalCharges DECIMAL(10,2),
	Churn VARCHAR(15)
	);

SELECT * FROM churn_data LIMIT 10;

--CHURN
SELECT churn, COUNT(customerid) FROM churn_data GROUP BY churn;

--GENDER WISE CHURN
SELECT gender,churn, COUNT(churn) FROM churn_data GROUP BY churn,gender;

--DEPENDENTS WISE CHURN
SELECT dependents,churn, COUNT(churn) FROM churn_data GROUP BY churn,dependents;

--PAYMENT METHOD WISE CHURN
SELECT paymentmethod,churn, COUNT(churn) FROM churn_data GROUP BY churn,paymentmethod;

--CONTRACT WISE CHURN
SELECT contract,churn, COUNT(churn) FROM churn_data GROUP BY churn,contract;

--PHONE SERVICE WISE CHURN
SELECT phoneservice,churn, COUNT(churn) FROM churn_data GROUP BY churn,phoneservice;

--INTERNET SERVICE WISE CHURN
SELECT internetservice,churn, COUNT(churn) FROM churn_data GROUP BY churn,internetservice;

--PARTNER WISE CHURN
SELECT partner,churn, COUNT(churn) FROM churn_data GROUP BY churn,partner;

--PAPERLESS PAYMENT WISE CHURN
SELECT paperlessbilling,churn, COUNT(churn) FROM churn_data GROUP BY churn,paperlessbilling;

--AVG TENURE
SELECT AVG(tenure) as AVERAGE_TENURE FROM churn_data;

--AVG MONTHLY CHARGES
SELECT AVG(monthlycharges) as AVERAGE_MONTHLY_CHARGES FROM churn_data;
