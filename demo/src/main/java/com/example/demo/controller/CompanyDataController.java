package com.example.demo.controller;

import com.example.demo.model.CompanyData;
import com.example.demo.model.CompanyDataTransaction;
import com.example.demo.service.CSVDataImporter;
import com.example.demo.service.CompanyDataService;
import com.example.demo.service.CompanyDataTransactionService;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping(value = "/api")
@Validated
@CrossOrigin(origins = "*")
public class CompanyDataController {

    private final CompanyDataService companyDataService;
    private final CompanyDataTransactionService companyDataTransactionService;

    public CompanyDataController(CompanyDataService companyDataService, CompanyDataTransactionService companyDataTransactionService, CSVDataImporter csvDataImporter) {
        this.companyDataService = companyDataService;
        this.companyDataTransactionService = companyDataTransactionService;
        companyDataService.fetchAndSaveCompany();
        csvDataImporter.importCSVToDatabase();
    }

    @GetMapping("/all")
    public ResponseEntity<List<CompanyData>> getAllData() {
        List<CompanyData> transactions = companyDataService.getAll();
        return ResponseEntity.ok(transactions);
    }

    @GetMapping("/{id}")
    public ResponseEntity<CompanyDataTransaction> getCompanyDataById(@PathVariable Long id) {
        return ResponseEntity.ok(companyDataTransactionService.getDataById(id));
    }

    @GetMapping("/companies")
    public ResponseEntity<List<String>> getAllCompanies() {
        return ResponseEntity.ok(companyDataService.getAllCodes());
    }

    @GetMapping("/transactions/{code}")
    public ResponseEntity<List<CompanyDataTransaction>> getDataByCode(@PathVariable String code) {
        List<CompanyDataTransaction> transactions = companyDataService.getTransactionByCompanyCode(code);
        return ResponseEntity.ok(transactions);
    }

    @GetMapping("/predict/{code}")
    public ResponseEntity<List<String>> getPredictionByCode(@PathVariable String code) {
        List<String> predictions = new ArrayList<>();
        try {
            predictions = companyDataTransactionService.getPrediction(code);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        return ResponseEntity.ok(predictions);
    }
}