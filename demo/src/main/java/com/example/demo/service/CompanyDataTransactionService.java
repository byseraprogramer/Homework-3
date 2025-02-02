package com.example.demo.service;

import com.example.demo.model.CompanyDataTransaction;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public interface CompanyDataTransactionService {
    List<CompanyDataTransaction> getAllData();
    CompanyDataTransaction getDataById(Long id);
    List<String> getPrediction(String companyId);
}