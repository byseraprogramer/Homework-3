package com.example.demo.service;

import com.example.demo.model.CompanyData;
import com.example.demo.model.CompanyDataTransaction;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public interface CompanyDataService {
    List<CompanyData> getAll();
    CompanyData getByCompanyCode(String code);
    List<CompanyDataTransaction> getTransactionByCompanyCode(String code);
    List<String> getAllCodes();
    void fetchAndSaveCompany();
}