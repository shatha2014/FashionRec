import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "limmen.fasttext_on_spark",
      scalaVersion := "2.11.8",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "fasttext_on_spark",
    libraryDependencies ++= Seq(
      scalaTest,
      mockito,
      sparkCore,
      sparkSql,
      sparkMlLib,
      sparkStreaming,
      scalaCsv,
      scallop,
      commonsMath
    )
  )
