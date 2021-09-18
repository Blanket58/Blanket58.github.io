---
layout: post
title: "本地使用Maven构建Drools项目"
date: 2021-09-18
description: "本地使用Maven构建Drools项目"
tag: Java
---

DROOLS (JBOSS RULES)为java语言开发的开源业务规则引擎，具有一个易于访问企业策略、易于调整以及易于管理的开源业务规则引擎，符合业内标准，速度快、效率高。业务分析师或审核人员可以利用它轻松查看业务规则，从而检验是否已编码的规则执行了所需的业务规则。

下面介绍如何使用IDEA来构建一个基于Maven的Drools项目。

1. 首先新建一个使用maven-archetype-quickstart架构的项目，注意java版本一定要小于等于15

   ![](/assets/2021-09-18-drools-with-maven-1.png)

2. Name与ArtifactId为项目名称

   ![](/assets/2021-09-18-drools-with-maven-2.png)

   注意这里的GroupId，它会变成项目建成后架构中该路径下的package名，斟酌填写

   ![](/assets/2021-09-18-drools-with-maven-3.png)

3. 打开项目根目录下的pom.xml文件，为项目添加maven依赖包，点击Alt+Insert会有如下弹窗

   ![](/assets/2021-09-18-drools-with-maven-4.png)

   依次添加下列依赖包：

   - org.drools:drools-core:7.45.0.Final
   - org.drools:drools-compiler:7.45.0.Final
   - org.drools:drools-templates:7.45.0.Final
   - org.drools:drools-decisiontables:7.45.0.Final
   - org.slf4j:slf4j-api:1.7.26
   - org.slf4j:slf4j-log4j12:1.7.31
   - junit:junit:4.13.2
   - com.alibaba:fastjson:1.2.75
   - com.cedarsoftware:json-io:4.12.0

4. 新建文件将项目补全至如下架构

   ![](/assets/2021-09-18-drools-with-maven-5.png)

   其中kmodule.xml中内容为：

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>

   <kmodule xmlns="http://jboss.org/kie/6.0.0/kmodule">
       <kbase name="rules" packages="org.example">
           <ksession name="ksession-rules"/>
       </kbase>
   </kmodule>
   ```

   rule.xls文件所在目录下为策略文件放置处，策略文件可以为xls文件，也可以为drl文件

5. 其中Demo.java文件为drools策略进行判断时使用的字段来自的类，我们这里填写一个示例

   ```java
   package org.example;


   public class Demo implements java.io.Serializable
   {
       private String x;
       private Integer y;

       public Demo() {
       }

       public String getX() {
           return x;
       }

       public void setX(String x) {
           this.x = x;
       }

       public Integer getY() {
           return y;
       }

       public void setY(Integer y) {
           this.y = y;
       }

       public Demo(String x, Integer y)
       {
           this.x = x;
           this.y = y;
       }

   }
   ```

6. 修改App.java文件内容，让drools项目可以在本地运行起来

   ```java
   package org.example;

   import com.alibaba.fastjson.JSONObject;
   import com.cedarsoftware.util.io.JsonWriter;
   import org.apache.log4j.BasicConfigurator;
   import org.apache.log4j.Level;
   import org.apache.log4j.Logger;
   import org.drools.decisiontable.SpreadsheetCompiler;
   import org.kie.api.KieServices;
   import org.kie.api.runtime.KieContainer;
   import org.kie.api.runtime.KieSession;
   import org.kie.internal.io.ResourceFactory;

   import java.io.BufferedWriter;
   import java.io.File;
   import java.io.FileWriter;
   import java.io.IOException;


   public class App {
       public static void runDrl() {
           BasicConfigurator.configure();
           Logger.getRootLogger().setLevel(Level.OFF);
           // 从工厂中获得KieServices实例
           KieServices kieServices = KieServices.Factory.get();
           // 从KieServices中获得KieContainer实例，其会加载kmodule.xml文件并load规则文件
           KieContainer kieContainer = kieServices.getKieClasspathContainer();
           // 建立KieSession到规则文件的通信管道
           KieSession kSession = kieContainer.newKieSession("ksession-rules");

           // 构建实体类对象
           Demo demo = new Demo();
           demo.setX("123");
           demo.setY(123);

           kSession.insert(demo);
           int count = kSession.fireAllRules();
           System.out.println("总共触发了: " + count + " 条规则");
           System.out.println(JsonWriter.formatJson(JSONObject.toJSON(demo).toString()));
       }

       public static void convertExcelToDrl() {
           SpreadsheetCompiler compiler = new SpreadsheetCompiler();
           // 最后一个参数是excel里 sheet 的名称
           String rules = compiler.compile(ResourceFactory.newClassPathResource("org.example" + File.separator + "rule.xls", "UTF-8"), "Sheet1");

           try {
               BufferedWriter out = new BufferedWriter(new FileWriter("src/main/resources/org.example/rule-excel.drl"));
               out.write(rules);
               out.close();
           } catch (IOException e) {
               e.printStackTrace();
           }
       }

       public static void main(String[] args) {
   //        convertExcelToDrl();
           runDrl();
       }
   }
   ```

   运行结束时程序会打印出共触发了哪些规则，并且将经drools决策修改后的Demo类中所有属性使用json的格式打印在屏幕上。注意到其中还有一个未使用的静态方法`convertExcelToDrl`，它的作用是当我们使用xls决策表时，单独运行该方法可以将决策表翻译并生成一个rule-excel.drl文件。**但需要明确的是，直接运行`runDrl`方法程序将直接从决策表中读取规则，无需手动先将其翻译为drl文件。**
