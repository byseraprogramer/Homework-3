package com.example.demo.service;

import com.example.demo.model.CompanyData;
import com.example.demo.model.CompanyDataTransaction;
import com.example.demo.model.CompanyDataTransaction;
import com.example.demo.repository.CompanyDataRepository;
import com.example.demo.repository.CompanyDataTransactionRepository;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@Service
public class CSVDataImporter {

    private final CompanyDataTransactionRepository companyTransactionRepository;

    private final CompanyDataRepository companyDataRepository;

    public CSVDataImporter(CompanyDataTransactionRepository companyTransactionRepository, CompanyDataRepository companyDataRepository) {
        this.companyTransactionRepository = companyTransactionRepository;
        this.companyDataRepository = companyDataRepository;
    }

    @Transactional
    public void importCSVToDatabase() {
        companyTransactionRepository.deleteAll();
        List<CompanyData> companies = companyDataRepository.findAll();

        for (CompanyData company : companies) {
            String filePath = String.format("src/main/resources/technical_analysis_%s.csv", company.getCompanyCode());
            try (CSVParser parser = CSVFormat.DEFAULT.withHeader().parse(new FileReader(filePath))) {
                for (CSVRecord record : parser) {
                    CompanyDataTransaction transaction = mapRecordToCompanyTransaction(record);
                    if (transaction != null) {
                        companyTransactionRepository.save(transaction);
                    }
                }

            } catch (IOException e) {
                throw new RuntimeException("Error reading CSV file: " + filePath, e);
            }
        }
    }

    private CompanyDataTransaction mapRecordToCompanyTransaction(CSVRecord record) {
        try {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");

            String code = record.get("NAME");
            String date = record.get("DATE");
            String lastPrice = record.get("PRICE OF LAST TRANSACTION IN mkd");
            String min = record.get("MIN");
            String max = record.get("MAX");
            String averagePrice = record.get("AVERAGE PRICE");
            String percentageChange = record.get("%CHANGE");
            String quantity = record.get("QUANTITY");
            String turnover = record.get("Turnover in BEST in mkd");
            String totalTurnover = record.get("TOTAL TURNOVER in mkd");
            String sma10 = record.get("SMA_10");
            String ema10 = record.get("EMA_10");
            String rsi = record.get("RSI");
            String cci = record.get("CCI");
            String k = record.get("%K");
            String d = record.get("%D");
            String signal = record.get("Signal");
            String cciSignal = record.get("CCI_Signal");
            String stoch_Signal = record.get("Stochastic_Signal");
            String ema20 = record.get("EMA_20");
            String wma10 = record.get("WMA_10");
            String sma20 = record.get("SMA_20");

            CompanyData company = companyDataRepository.getReferenceById(code);
            CompanyDataTransaction companyTransaction = new CompanyDataTransaction(company, date, lastPrice, min, max, averagePrice, percentageChange, quantity, turnover, totalTurnover, sma10, ema10, rsi, cci, k, d, signal, cciSignal, stoch_Signal, ema20, wma10, sma20);

            company.addTransaction(companyTransaction);

            return companyTransaction;
        } catch (IllegalArgumentException e) {
            System.err.println("Skipping invalid record: " + record);
            return null;
        }
    }

    private Double parseDouble(String value) {
        return value == null || value.isEmpty() ? null : Double.valueOf(value);
    }

    private Long parseLong(String value) {
        return value == null || value.isEmpty() ? null : Long.valueOf(value);
    }

    private Boolean parseBoolean(String value) {
        return value != null && value.equalsIgnoreCase("true");
    }
}