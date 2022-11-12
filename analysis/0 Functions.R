numDesignMatColsFromDataset <- function(df) {
  num <- 0
  for (name in names(df)) {
    if (is.factor(df[[name]]))
      num <-
        num + length(unique(df[[name]][complete.cases(df[[name]])])) - 1
    else
      num <- num + 1
  }
  num
}

setColTypesForModeling <- function(df) {
  for (name in names(df)) {
    col_type <- typeof(df[, name])
    if (col_type == 'character')
      df[, name] <- as.factor(df[, name])
    else if (col_type == 'logical')
      df[, name] <- as.factor(df[, name])
    else if (col_type == 'integer')
      df[, name] <- as.numeric(df[, name])
  }
  df
}

updateDeckSideNum <- function (df) {
  df$Deck <-
    as.character(sapply(df$Cabin, function(x) {
      ifelse(is.na(x), NA, strsplit(x, '/')[[1]][1])
    }))
  df$Num <-
    as.integer(sapply(df$Cabin, function(x) {
      ifelse(is.na(x), NA, strsplit(x, '/')[[1]][2])
    }))
  df$Side <-
    as.character(sapply(df$Cabin, function(x) {
      ifelse(is.na(x), NA, strsplit(x, '/')[[1]][3])
    }))
  df
}

updateSpending <- function(df) {
  rsna <- is.na(df$RoomService)
  fcna <- is.na(df$FoodCourt)
  smna <- is.na(df$ShoppingMall)
  sna <- is.na(df$Spa)
  vrdna <- is.na(df$VRDeck)
  allna <- rsna & fcna & smna & sna & vrdna
  df$Spending <-
    ifelse(
      allna,
      NA,
      ifelse(rsna, 0, df$RoomService) + ifelse(fcna, 0, df$FoodCourt) + ifelse(smna, 0, df$ShoppingMall) + ifelse(sna, 0, df$Spa) + ifelse(vrdna, 0, df$VRDeck)
    )
  df$HasSpent <- df$Spending > 0
  df
}

updateSpendingOfCryo <- function(df) {
  df[df$CryoSleep %in% T, c("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")] <-
    0
  df$CryoSleep[df$HasSpent %in% T] <- F
  df
}

updateSpendingOfChildren <- function(df) {
  df[!is.na(df$Age) &
       df$Age <= 12, c("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")] <-
    0
  df <- updateSpending(df)
  df
}

updateSideFromGID <- function(df) {
  uni_GID <- unique(ship$GID)
  uni_side <-
    sapply(uni_GID, function(i) {
      df$Side[!is.na(df$Side) & df$GID == i][1]
    })
  dat <- data.frame(uni_GID, uni_side)
  for (i in uni_GID) {
    df$Side[is.na(df$Side) &
              df$GID == i] <-  dat$uni_side[dat$uni_GID == i]
  }
  df
}

updateHomePlanetFromSurname <- function(df) {
  earth_sur <-
    unique(df$LastName[!is.na(df$LastName) &
                         !is.na(df$HomePlanet) & df$HomePlanet == "Earth"])
  mars_sur <-
    unique(df$LastName[!is.na(df$LastName) &
                         !is.na(df$HomePlanet) & df$HomePlanet == "Mars"])
  europa_sur <-
    unique(df$LastName[!is.na(df$LastName) &
                         !is.na(df$HomePlanet) & df$HomePlanet == "Europa"])
  uni_surname <- c(earth_sur, mars_sur, europa_sur)
  dat <-
    data.frame("home" = factor(c(
      rep("Earth", length(earth_sur)),
      rep("Mars", length(mars_sur)),
      rep("Europa", length(europa_sur))
    )),
    "LastName" = uni_surname)
  for (i in uni_surname) {
    df$HomePlanet[is.na(df$HomePlanet) &
                    !is.na(df$LastName) &
                    df$LastName == i] <- dat$home[dat$LastName == i]
  }
  df
}

updateHomePlanetFromDeck <- function(df) {
  df[!is.na(df$Deck) & df$Deck == "G", "HomePlanet"] <- "Earth"
  df[!is.na(df$Deck) &
       df$Deck %in% c("A", "B", "C", "T"), "HomePlanet"] <- "Europa"
  df
}

updateVIPFromHomePlanet <- function(df) {
  df$VIP[!is.na(df$HomePlanet) & df$HomePlanet == 'Earth'] <- F
  df
}

savePredictions <- function(passenger_ids, predictions, filename) {
  write.table(
    data.frame(
      PassengerId = passenger_ids,
      Transported = as.character(ifelse(predictions == 'TRUE', 'True', 'False'))
    ),
    file = paste0('submissions/', filename, '.csv'),
    quote = F,
    row.names = F,
    col.names = c("PassengerId", "Transported"),
    sep = ","
  )
}
