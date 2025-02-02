package com.example.demo.service.impl;

import com.example.demo.model.CompanyData;
import com.example.demo.model.CompanyDataTransaction;
import com.example.demo.repository.CompanyDataRepository;
import com.example.demo.service.CompanyDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class CompanyDataServiceImpl implements CompanyDataService {
    @Autowired
    private final CompanyDataRepository companyDataRepository;

    public CompanyDataServiceImpl(CompanyDataRepository companyDataRepository) {
        this.companyDataRepository = companyDataRepository;
    }

    private final RestTemplate restTemplate = new RestTemplate();

    @Override
    public List<CompanyData> getAll() {
        return companyDataRepository.findAll();
    }

    @Override
    public CompanyData getByCompanyCode(String code) {
        return companyDataRepository.getReferenceById(code);
    }

    @Override
    public List<CompanyDataTransaction> getTransactionByCompanyCode(String code) {
        return companyDataRepository.getReferenceById(code).getTransactions();
    }

    @Override
    public List<String> getAllCodes() {
        return companyDataRepository.findAll().stream().map(CompanyData::getCompanyCode).collect(Collectors.toList());
    }

    public void fetchAndSaveCompany() {
        List<String> codes = restTemplate.getForObject("http://127.0.0.1:5000/api/companies", List.class);
        assert codes != null;
        for (String code : codes) {
            CompanyData company = new CompanyData(code);
            companyDataRepository.save(company);
        }

//    @Override
//    public List<CompanyData> getAllCompanyData() {
//        return companyDataRepository.findAll();
//    }
//
//    @Override
//    public CompanyData getCompanyDataById(Long id) {
//        return companyDataRepository.findById(id).orElse(null);
//    }
//
//    @Override
//    public List<CompanyData> getDataByCode(String code) {
//        return companyDataRepository.findAll().stream().filter(companyData -> companyData.getCompanyCode().equals(code)).collect(Collectors.toList());
//    }
    }
}