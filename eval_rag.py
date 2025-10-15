import os
from rag_chatbot import get_answer
from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"

client = Client()

eval_data = [
    {
        "question": "What is a Non-Banking Financial Company (NBFC)?",
        "expected_answer": "A Non-Banking Financial Company (NBFC) is a company registered under the Companies Act, 1956 or Companies Act, 2013, and engaged in the business of loans and advances, acquisition of shares/stocks/bonds/debentures/securities issued by Government or local authority or other marketable securities of a like nature, leasing, hire-purchase, etc., as their principal business, but does not include any institution whose principal business is that of agriculture activity, industrial activity, purchase or sale of any goods (other than securities) or providing any services and sale/purchase/construction of immovable property. A non-banking institution which is a company and has principal business of receiving deposits under any scheme or arrangement in one lump sum or in installments by way of contributions or in any other manner, is also a non-banking financial company (Residuary non-banking company)."
    },
    {
        "question": "What does conducting financial activity as “principal business” mean?",
        "expected_answer": "Financial activity as principal business is when a company’s financial assets constitute more than 50 per cent of the total assets (netted off by intangible assets) and income from financial assets constitute more than 50 per cent of the gross income. A company which fulfils both these criteria needs to get registered as NBFC with the Reserve Bank. The term 'principal business' has not been defined in the Reserve Bank of India Act, 1934. Hence, the Reserve Bank has defined it vide Press Release 1998-99/1269 dated April 08, 1999 so as to ensure that only companies predominantly engaged in financial activity get registered with it and are regulated and supervised by it. Hence, if there are companies engaged in agricultural operations, industrial activity, purchase and sale of goods, providing services or purchase, sale or construction of immovable property as their principal business and are doing some financial business in a small way, they will not be regulated by the Reserve Bank. Interestingly, this test is popularly known as 50-50 test and is applied to determine whether or not a company is into financial business."
    },
    {
        "question": "NBFCs are doing functions similar to banks. What is the difference between banks and NBFCs?",
        "expected_answer": "Banks and NBFCs are different entities subject to different statutory and regulatory requirements. However, NBFCs lend and make investments and hence these activities are akin to that of banks. The major differences between banks and NBFCs are given below:\n\ni. NBFCs cannot accept demand deposits;\n\nii. NBFCs do not form part of the payment and settlement system and cannot issue cheques drawn on itself;\n\niii. Deposit insurance facility of Deposit Insurance and Credit Guarantee Corporation (DICGC) is not available to depositors of deposit taking NBFCs."
    },
    {
        "question": "Is it necessary that every NBFC should be registered with the Reserve Bank?",
        "expected_answer": "In terms of Section 45-IA of the RBI Act, 1934, no NBFC can commence or carry on business of a non-banking financial institution without a) obtaining a certificate of registration from the Reserve Bank and without having a Net Owned Funds (NOF) of ₹10 crore with effect from October 01, 2022 (NBFCs seeking registration shall have NOF of ₹10 crore ab initio, and existing NBFCs have timeline upto March 31, 2027 to attain NOF of ₹10 crore). However, in terms of the powers conferred upon the Reserve Bank, to obviate dual regulation, certain categories of NBFCs which are regulated by other regulators are exempted from the requirement of registration with the Reserve Bank viz., Alternative Investment Fund/ Merchant Banking companies/ Stock broking companies registered with SEBI; Insurance Company holding a valid Certificate of Registration issued by IRDA; Nidhi companies as notified under Section 620A of the Companies Act, 1956; Chit companies doing the business of chits as defined in clause (b) of Section 2 of the Chit Funds Act, 1982; Stock Exchange or a Mutual Benefit company, etc."
    },
    {
        "question": "What are the requirements for registration with the Reserve Bank?",
        "expected_answer": "A ‘company’ desirous of commencing the business of non-banking financial institution as defined under Section 45 I(a) of the RBI Act, 1934 should comply with the following:\n\ni. It should be a company incorporated under Section 3 of the companies Act, 1956 or corresponding Section under the Companies Act, 2013;\n\nii. It should have a minimum net owned fund of ₹10 crore. (The minimum net owned fund requirements for specialized NBFCs are NBFC-Infrastructure Finance Company (NBFC-IFC) – ₹300 crore; Infrastructure Debt Fund – NBFC (IDF-NBFC) – ₹300 crore; Mortgage Guarantee Company (MGC) – ₹100 crore; Housing Finance Company (HFC) – ₹20 crore, Standalone Primary Dealers (SPDs) which undertake only the core activities – ₹150 crore and SPDs which also undertake non-core activities – ₹250 crore; NBFC-AA – ₹2 crore; and NBFC-P2P – ₹2 crore)."
    },
    {
        "question": "What is the procedure for application to the Reserve Bank for Registration?",
        "expected_answer": "The applicant company is required to apply online on https://pravaah.rbi.org.in and also submit a physical copy of the application along with the necessary documents as per the process prescribed by the Reserve Bank vide its Press Release 2015-2016/2935 dated June 17, 2016 to the Chief General Manager-in-Charge, Department of Regulation, Reserve Bank of India, Central Office, 2nd Floor, Main Office Building, Shahid Bhagat Singh Marg, Fort, Mumbai-400 001."
    },
    {
        "question": "What are the essential documents required to be submitted along with the application form to the Reserve Bank?",
        "expected_answer": "The application form and an indicative checklist of the documents required to be submitted along with the application is available on Reserve Bank’s website under NBFC Forms."
    },
    {
        "question": "What is Scale Based Regulatory Framework or SBR Framework for NBFCs?",
        "expected_answer": "Over the years, the NBFC sector had evolved considerably in terms of size, complexity, and interconnectedness within the financial sector and hence there was a need to align the regulatory framework for NBFCs keeping in view their changing risk profile. Accordingly, the Reserve Bank has implemented a Scale-Based Regulatory Framework or SBR Framework for regulation of NBFCs w.e.f. October 01, 2022. The SBR Framework which is based on the principle of proportionality takes into account various factors like size, activity, complexity, interconnectedness, etc., within the financial sector for categorising NBFCs into various layers. The degree of regulations increases as one moves from lower to higher layers. SBR Framework classifies NBFCs into four layers. NBFCs in the lowest layer shall be known as NBFC – Base Layer (NBFC-BL). NBFCs in middle layer and upper layer shall be known as NBFC – Middle Layer (NBFC-ML) and NBFC – Upper Layer (NBFC-UL) respectively and are considered to be systemically significant. The Top Layer is ideally expected to be empty and will be known as NBFC - Top Layer (NBFC-TL) which will be populated only if the Reserve Bank is of the opinion that there is a substantial increase in the potential systemic risk from specific NBFCs in the Upper Layer."
    },
    {
        "question": "Does the Reserve Bank regulate all financial companies?",
        "expected_answer": "No, the Reserve Bank does not regulate all financial companies. Depending upon the nature of activities, the financial companies may fall under the regulatory purview of other Regulators like SEBI, IRDAI, Government, etc. To name a few, the Merchant Banking Companies/Alternative Investment Fund Company/stock-exchanges/stock brokers/sub-brokers are regulated by Securities and Exchange Board of India, and Insurance companies are regulated by Insurance Regulatory and Development Authority. Similarly, Chit Fund Companies are regulated by the respective State Governments and Nidhi Companies are regulated by Ministry of Corporate Affairs, Government of India. Companies that do financial business but are regulated by other regulators are given specific exemption by the Reserve Bank from its regulatory requirements for avoiding duality of regulation. The categories of NBFCs which are exempted from certain provisions of the RBI Act, 1934 are specified in the ‘Master Direction - Exemptions from the provisions of RBI Act, 1934 dated August 25, 2016."
    },
    {
        "question": "What are the different types/categories of NBFCs registered with the Reserve Bank?",
        "expected_answer": "NBFCs are categorized (a) in terms of the type of liabilities into Deposit and Non-Deposit accepting NBFCs; (b) regulatory structure of NBFCs under Scale Based Regulation into NBFC-Base Layer, NBFC-Middle Layer, NBFC-Upper Layer, and NBFC-Top Layer (as detailed in FAQ no.8 above); and (c) by the kind of activity they conduct. Based on the type of activities they conduct, the different types of NBFCs are as follows: I. Investment and Credit Company (ICC), II. Housing Finance Company (HFC), III. Infrastructure Finance Company (IFC), IV. Infrastructure Debt Fund (IDF-NBFC), V. Core Investment Company (CIC), VI. Micro Finance Institution (NBFC-MFI), VII. Non-Banking Financial Company – Factors (NBFC-Factors), VIII. Mortgage Guarantee Companies (MGC), IX. Standalone Primary Dealers (SPDs), X. Non-Operative Financial Holding Company (NOFHC), XI. NBFC – Account Aggregator (NBFC-AA), XII. NBFC – Peer to Peer Lending Platform (NBFC-P2P)."
    },
    {
        "question": "What are the powers of the Reserve Bank with regard to 'Non-Bank Financial Companies’?",
        "expected_answer": "The Reserve Bank has been empowered under the RBI Act 1934 to register, determine policy, issue directions, inspect, regulate, supervise and exercise surveillance over NBFCs that fulfil the principal business criteria or 50-50 criteria of principal business. The Reserve Bank can penalize NBFCs for violating the provisions of the RBI Act or the directions or orders issued by the Reserve Bank under RBI Act. The penal action may also include cancellation of the Certificate of Registration issued to the NBFC."
    },
    {
        "question": "What action can be taken against persons/financial companies making false claim of being regulated by the Reserve Bank?",
        "expected_answer": "It is illegal for any person/ entity/ financial company to make a false claim of being regulated by the Reserve Bank to mislead the public to collect deposits and is liable for penal action under the Law. Information in this regard may be forwarded to the nearest office of the Reserve Bank and the Police."
    },
    {
        "question": "What action is taken if financial companies which are lending or making investments as their principal business do not obtain a Certificate of Registration from the Reserve Bank?",
        "expected_answer": "If companies that are required to be registered with the Reserve Bank as NBFCs, are found to be conducting non-banking financial activity, such as, lending, investment or deposit acceptance as their principal business, without obtaining Certificate of Registration from the Reserve Bank, the same would be treated as contravention of the provisions of the RBI Act, 1934 and would invite penal action viz., penalty or fine or even prosecution in a Court of Law. If members of public come across any entity which undertakes non-banking financial activity but does not figure in the list of authorized NBFCs on the Reserve Bank’s website, they should inform the nearest Regional Office of the Reserve Bank, for appropriate action to be taken for contravention of the provisions of the RBI Act, 1934."
    },
    {
        "question": "Where can one find list of Registered NBFCs and instructions issued to NBFCs?",
        "expected_answer": "The list of registered NBFCs is available on the web site of Reserve Bank (www.rbi.org.in) under ‘Regulation → Non-Banking’. Further, the instructions issued to NBFCs from time to time through circulars and/ or master directions are hosted on the Reserve Bank’s website under ‘Notifications’, and some instructions are issued through Official Gazette notifications and press releases as well."
    },
    {
        "question": "What are the regulations prescribed by the Reserve Bank for NBFCs?",
        "expected_answer": "As part of regulatory framework prescribed by the Reserve Bank for NBFCs, the Reserve Bank prescribes prudential regulations viz., capital adequacy/ leverage, provisioning, corporate governance framework, etc.; conduct of business regulations viz., KYC/ AML regulations, fair practices code, etc.; and other miscellaneous regulations to ensure that NBFCs are financially sound and follow transparency in their operations. The regulations for NBFCs are contained in various master directions and notifications/ circulars issued from time to time, and are available on the website of the Reserve Bank (www.rbi.org.in) under ‘notifications’"
    }
]



evaluator = load_evaluator(
    "labeled_criteria",
    criteria="correctness",
    llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
)

dataset = client.create_dataset(
    dataset_name="RBI RAG Eval - Gemini",
    description="Evaluation of NBFC RAG chatbot answers using Gemini via LangSmith."
)


results = []

for i, item in enumerate(eval_data, 1):
    question = item["question"]
    reference = item["expected_answer"]

    try:
        prediction = get_answer(question)
        eval_result = evaluator.evaluate_strings(
            input=question,
            prediction=prediction,
            reference=reference
        )
    except Exception as e:
        prediction = f"Error: {e}"
        eval_result = {"score": 0, "reason": str(e)}

    # Save result
    results.append({
        "question": question,
        "predicted": prediction,
        "reference": reference,
        "evaluation": eval_result
    })

    example = client.create_example(
        dataset_id=dataset.id,
        inputs={"question": question},
        outputs={
            "prediction": prediction,
            "reference": reference,
            "evaluation": eval_result
        },
        metadata={"source": "RAG chatbot"}
    )


scores = [r["evaluation"].get("score", 0) for r in results if "evaluation" in r]
avg_score = sum(scores) / len(scores) if scores else 0

print(f"\nEvaluation completed successfully!")
print(f"Average Correctness Score: {avg_score:.3f}")